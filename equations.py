from collections.abc import Mapping

import numpy as np

# name, F coefficient of BS², F coefficient of B,
# D/4 coefficient of BS², D/4 coefficient of B, parameter count
SPEC = [
    ('conv1',2352,0,11,0,4704), ('relu1',0,0,16,0,0),
    ('pool',0,0,10,0,0),
    ('conv2',6400,0,6,0,51200), ('relu2',0,0,8,0,0),
    ('conv3',2304,0,6,0,73728), ('relu3',0,0,4,0,0),
    ('conv4',1024,0,6,0,32768), ('relu4',0,0,8,0,0),
    ('conv5',4608,0,5,0,589824), ('relu5',0,0,2,0,0),
    ('conv6',1024,0,3,0,131072), ('relu6',0,0,4,0,0),
    ('avgpool',2,0,2,512,0),
    ('fc1',0,262400,0,768,131328), ('relu7',0,0,0,512,0),
    ('fc2',0,51300,0,356,25700),
]
NAMES = [r[0] for r in SPEC]


def layer_costs(image_size, batch):
    s,b=np.broadcast_arrays(np.asarray(image_size,dtype=float),np.asarray(batch,dtype=float))
    x=b*s*s
    f=np.stack([a*x+c*b for _,a,c,_,_,_ in SPEC],axis=-1)
    d=np.stack([4*(a*x+c*b+p) for _,_,_,a,c,p in SPEC],axis=-1)
    return f,d


PARAMETER_COUNT = 1_040_324
FP32_BYTES = 4


def _inputs(image_size, batch):
    # Convert before arithmetic to avoid overflow with integer NumPy dtypes.
    size, batch = np.broadcast_arrays(
        np.asarray(image_size, dtype=np.float64),
        np.asarray(batch, dtype=np.float64),
    )
    if (np.any(~np.isfinite(size)) or np.any(size <= 0)
            or np.any(size % 16 != 0)):
        raise ValueError("image_size must contain positive multiples of 16")
    if (np.any(~np.isfinite(batch)) or np.any(batch <= 0)
            or np.any(batch % 1 != 0)):
        raise ValueError("batch must contain positive integers")
    return size, batch


def _result(value):
    value = np.asarray(value, dtype=np.float64)
    return value.item() if value.ndim == 0 else value


def _parameter(theta, name):
    if not isinstance(theta, Mapping):
        raise TypeError("theta must be a mapping of named scalar parameters")
    if name not in theta:
        raise ValueError(f"Missing parameter: {name}")
    value = np.asarray(theta[name], dtype=np.float64)
    if value.ndim != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value.item()


def flops(image_size, batch):
    """FLOPs, using 2 FLOPs/MAC; includes Linear bias and global averaging.

    Comparisons in ReLU and MaxPool are excluded.
    """
    size, batch = _inputs(image_size, batch)
    return _result(17_714 * batch * size**2 + 313_700 * batch)


def memory(image_size, batch):
    """Bytes: parameters + input + summed activation storage, without freeing.

    In-place ReLU/Flatten add no storage. Excludes workspace and allocator
    rounding; this is neither the measured peak nor a guaranteed upper bound.
    """
    size, batch = _inputs(image_size, batch)
    return _result(FP32_BYTES * (
        PARAMETER_COUNT + 26 * batch * size**2 + 868 * batch
    ))


def bytes_moved(image_size, batch):
    """Idealized bytes: one input/parameter read and one output write/operator.

    Parameters are read once per batch. ReLU reads and writes in-place;
    Flatten has zero traffic. This is not a worst-case DRAM traffic estimate.
    """
    size, batch = _inputs(image_size, batch)
    return _result(FP32_BYTES * (
        PARAMETER_COUNT + 91 * batch * size**2 + 2_148 * batch
    ))


def latency(image_size, batch, theta):
    """Seconds: sum_i max(tau, F_i/C, D_i/W) over 17 non-view operators.

    Final schema: tau (seconds/operator), C (FLOPs/s), W (bytes/s).
    Legacy mappings with t0 retain the aggregate baseline for comparison.

    theta is a mapping with positive scalar keys:
      t0: effective per-forward overhead, seconds;
      C: effective compute throughput, FLOPs/second;
      W: effective memory bandwidth, bytes/second.
    These parameters must be calibrated, not taken as hardware specifications.
    """
    if "tau" in theta:
        tau = _parameter(theta, "tau")
        compute = _parameter(theta, "C")
        bandwidth = _parameter(theta, "W")
        size, batch = _inputs(image_size, batch)
        f, d = layer_costs(size, batch)
        return _result(np.maximum(tau, np.maximum(f / compute, d / bandwidth)).sum(axis=-1))
    t0 = _parameter(theta, "t0")
    compute = _parameter(theta, "C")
    bandwidth = _parameter(theta, "W")
    return _result(np.maximum(t0, np.maximum(
        flops(image_size, batch) / compute,
        bytes_moved(image_size, batch) / bandwidth,
    )))


def energy(image_size, batch, theta_energy):
    """Whole-GPU joules: p0*T + p1*max(T-17*tau, 0).

    Final schema: tau, C, W, p0, p1. Both powers are positive in watts;
    p0 is an effective overhead-regime power, not a measured idle power.
    Legacy q*T remains available for the archived aggregate baseline.
    No idle-energy subtraction.

    theta_energy contains t0, C, W as in latency, plus q: positive effective
    average power in watts. Fit q after fixing the calibrated time parameters.
    """
    if "tau" in theta_energy:
        p0 = _parameter(theta_energy, "p0")
        p1 = _parameter(theta_energy, "p1")
        tau = _parameter(theta_energy, "tau")
        t = latency(image_size, batch, theta_energy)
        # U is time beyond the fitted launch/overhead floor, never negative.
        u = np.maximum(0.0, t - 17 * tau)
        return _result(p0 * t + p1 * u)
    power = _parameter(theta_energy, "q")
    return _result(power * latency(image_size, batch, theta_energy))
