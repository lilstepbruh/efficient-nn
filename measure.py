import argparse
import csv
import ctypes
import gc
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

try:
    from .models import SmallCNN
    from .equations import memory
except ImportError:
    from models import SmallCNN
    from equations import memory

BASE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
FIELDS = [
    'S', 'B', 'is_validation', 'status', 'latency', 'memory', 'energy',
    'energy_status', 'latency_p25', 'latency_p75', 'memory_predicted',
    'free_bytes_before_input', 'reserved_peak_bytes', 'energy_per_repeat_j',
    'energy_passes', 'energy_durations_s', 'error',
]


def make_grid(seed=42):
    rng = np.random.default_rng(seed)
    extra_sizes = sorted(map(int, rng.choice(
        [s for s in range(32, 513, 16) if s not in BASE_SIZES], 4, replace=False)))
    extra_batches = sorted(map(int, rng.choice(
        [b for b in range(1, 257) if b not in BASE_BATCHES], 3, replace=False)))
    configs = [dict(S=s, B=b, is_validation=s not in BASE_SIZES or b not in BASE_BATCHES)
               for s in sorted(BASE_SIZES + extra_sizes)
               for b in sorted(BASE_BATCHES + extra_batches)]
    # Randomize order to reduce correlation between shape and thermal drift.
    order = rng.permutation(len(configs))
    return dict(seed=seed, extra_sizes=extra_sizes, extra_batches=extra_batches,
                configurations=[configs[int(i)] for i in order])


class EnergyCounter:
    """Read whole-device accumulated energy in mJ using system NVML."""
    def __init__(self, uuid):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        self.check(self.lib.nvmlInit_v2())
        try:
            self.handle = ctypes.c_void_p()
            self.check(self.lib.nvmlDeviceGetHandleByUUID(str(uuid).encode(), ctypes.byref(self.handle)))
            self.read_mj()  # Fail explicitly if energy counters are unsupported.
        except Exception:
            self.close()
            raise

    def check(self, code):
        if code:
            message = self.lib.nvmlErrorString(code).decode()
            raise RuntimeError(f'NVML: {message} ({code})')

    def read_mj(self):
        value = ctypes.c_ulonglong()
        self.check(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(value)))
        return value.value

    def driver(self):
        value = ctypes.create_string_buffer(96)
        self.check(self.lib.nvmlSystemGetDriverVersion(value, len(value)))
        return value.value.decode()

    def close(self):
        self.lib.nvmlShutdown()


def forward_sync(model, x):
    output = model(x)
    torch.cuda.synchronize()
    del output


@torch.inference_mode()
def measure_one(model, counter, config, args):
    s, b = config['S'], config['B']
    row = {**config, 'status': 'OK', 'energy_status': 'OK',
           'memory_predicted': int(memory(s, b)), 'error': ''}
    row['free_bytes_before_input'] = torch.cuda.mem_get_info()[0]
    generator = torch.Generator(device='cuda').manual_seed(args.seed + s * 257 + b)
    x = torch.randn(b, 3, s, s, device='cuda', dtype=torch.float32, generator=generator)
    for _ in range(args.warmup):
        forward_sync(model, x)

    # Peak allocated includes the already-live parameters and input, plus a
    # single forward's activations/workspace/output. No previous output is live.
    torch.cuda.reset_peak_memory_stats()
    forward_sync(model, x)
    row['memory'] = torch.cuda.max_memory_allocated()
    row['reserved_peak_bytes'] = torch.cuda.max_memory_reserved()

    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        forward_sync(model, x)
        samples.append(time.perf_counter() - start)
    row['latency'] = statistics.median(samples)
    row['latency_p25'], row['latency_p75'] = np.percentile(samples, [25, 75]).tolist()

    energies, passes, durations = [], [], []
    for _ in range(args.energy_repeats):
        torch.cuda.synchronize()
        e0 = counter.read_mj()
        start = time.perf_counter()
        count = 0
        elapsed = 0.0
        # Same per-forward synchronization as latency, rather than comparing
        # isolated latency with a different asynchronous throughput workload.
        while elapsed < args.energy_seconds:
            forward_sync(model, x)
            count += 1
            elapsed = time.perf_counter() - start
        e1 = counter.read_mj()
        if e1 <= e0:
            raise RuntimeError('Energy counter did not advance monotonically')
        energies.append((e1 - e0) / (1000 * count))
        passes.append(count)
        durations.append(elapsed)
    row['energy'] = statistics.median(energies)
    row['energy_per_repeat_j'] = json.dumps(energies)
    row['energy_passes'] = json.dumps(passes)
    row['energy_durations_s'] = json.dumps(durations)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=21)
    parser.add_argument('--energy-repeats', type=int, default=3)
    parser.add_argument('--energy-seconds', type=float, default=1.0)
    parser.add_argument('--smoke', action='store_true', help='Three diagnostic points; use a separate output directory')
    parser.add_argument('--grid-only', action='store_true', help='Save grid and split without accessing CUDA')
    parser.add_argument('--resume', action='store_true', help='Continue identical recorded protocol without duplicate rows')
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.energy_repeats) < 1 or not np.isfinite(args.energy_seconds) or args.energy_seconds <= 0:
        parser.error('Repeat counts and energy duration must be positive')
    grid = make_grid(args.seed)
    if args.smoke:
        grid['configurations'] = [dict(S=s, B=b, is_validation=False)
                                  for s, b in [(32, 1), (224, 8), (512, 256)]]
        grid['diagnostic_only'] = True
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    grid_path = out / 'grid.json'
    if grid_path.exists() and json.loads(grid_path.read_text()) != grid:
        parser.error('Output directory contains a different grid; choose another directory')
    grid_path.write_text(json.dumps(grid, indent=2) + '\n')
    configs = grid['configurations']
    print(f'Grid: {len(configs)} points; extra S={grid["extra_sizes"]}, B={grid["extra_batches"]}', flush=True)
    if args.grid_only:
        return
    csv_path = out / 'measurements.csv'
    meta_path = out / 'measurement_protocol.json'
    if (csv_path.exists() or meta_path.exists()) and not args.resume:
        parser.error('Existing measurements/protocol; use --resume or another directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run with GPU access in the existing environment')
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    counter = EnergyCounter(props.uuid)
    try:
        source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                         for name in ('models.py', 'equations.py', 'measure.py')}
        settings = dict(seed=args.seed, warmup=args.warmup, repeats=args.repeats,
                        energy_repeats=args.energy_repeats, energy_seconds=args.energy_seconds,
                        smoke=args.smoke, torch=torch.__version__, python=platform.python_version(),
                        numpy=np.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                        driver=counter.driver(), gpu=props.name, gpu_uuid=str(props.uuid),
                        total_memory_bytes=props.total_memory, source_sha256=source_hashes,
                        tf32=False, cudnn_benchmark=False, dtype='float32')
        if meta_path.exists():
            if json.loads(meta_path.read_text())['settings'] != settings:
                parser.error('Resume settings/source/environment differ from saved protocol')
        elif csv_path.exists():
            parser.error('Cannot resume CSV without measurement_protocol.json')
        else:
            meta_path.write_text(json.dumps(dict(
                started_utc=datetime.now(timezone.utc).isoformat(), settings=settings,
                latency_method='median perf_counter around model forward plus CUDA synchronize; inputs already resident',
                memory_method='single warmed forward, reset_peak_memory_stats with model and input alive',
                energy_method='NVML whole-GPU cumulative mJ difference / passes / 1000; median of repeated intervals, synchronized each pass; no idle subtraction',
                validation_rule='any extra S or extra B; base grid only is calibration',
                limitations='energy includes other GPU activity and host-loop gaps; counter resolution and thermal drift may affect measurements',
            ), indent=2) + '\n')
        completed = set()
        if csv_path.exists():
            with csv_path.open(newline='') as f:
                for row in csv.DictReader(f):
                    key = (int(row['S']), int(row['B']))
                    if key in completed:
                        raise RuntimeError('Duplicate configurations in existing CSV')
                    completed.add(key)
        model = SmallCNN().float().cuda().eval()
        with csv_path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                writer.writeheader()
            for i, config in enumerate(configs, 1):
                if (config['S'], config['B']) in completed:
                    continue
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    row = measure_one(model, counter, config, args)
                except torch.cuda.OutOfMemoryError as exc:
                    row = {**config, 'status': 'OOM', 'memory': 'OOM', 'energy_status': 'not_measured',
                           'memory_predicted': int(memory(config['S'], config['B'])), 'error': str(exc)}
                # An exception other than OOM stops the run instead of silently
                # turning broken energy/timing measurements into usable data.
                writer.writerow(row)
                f.flush()
                status = row['status']
                if status == 'OK':
                    status += f" {row['latency']*1000:.3f} ms, {row['memory']/2**20:.1f} MiB, {row['energy']:.6f} J"
                print(f"[{i}/{len(configs)}] S={config['S']} B={config['B']}: {status}", flush=True)
        print(f'Saved: {csv_path}', flush=True)
    finally:
        counter.close()


if __name__ == '__main__':
    main()
