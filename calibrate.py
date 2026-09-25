import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

try:
    from .equations import layer_costs as costs, NAMES
except ImportError:
    from equations import layer_costs as costs, NAMES

try:
    from .equations import flops, memory, bytes_moved, latency, energy
except ImportError:
    from equations import flops, memory, bytes_moved, latency, energy


def read_measurements(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    result = []
    seen = set()
    for row in rows:
        if row['is_validation'] not in ('True', 'False'):
            raise ValueError('Invalid split flag')
        key = (int(row['S']), int(row['B']))
        if key in seen:
            raise ValueError('Duplicate configuration')
        seen.add(key)
        if row['status'] == 'OOM':
            continue
        if row['status'] != 'OK' or row['energy_status'] != 'OK':
            raise ValueError('Unexpected measurement status')
        item = dict(S=key[0], B=key[1], is_validation=row['is_validation'] == 'True')
        for name in ('latency', 'memory', 'energy'):
            item[name] = float(row[name])
            if not np.isfinite(item[name]) or item[name] <= 0:
                raise ValueError(f'Invalid {name}')
        result.append(item)
    return result


def fit_parameters(training, starts=32, seed=42, layer_path=None):
    if not training or any(row['is_validation'] for row in training):
        raise ValueError('Fit requires non-empty calibration-only rows')
    if layer_path is None:
        layer_path=Path(__file__).parent/'results'/'layers'/'measurements.csv'
    with Path(layer_path).open() as file:
        layers=list(csv.DictReader(file))
    train_keys={(r['S'],r['B']) for r in training}
    # Independently calibrate W from large, memory-only ReLU operations.
    # Logical traffic >=64 MiB corresponds to inputs >=32 MiB.
    references=[r for r in layers if r['is_validation']=='False'
                and (int(r['S']),int(r['B'])) in train_keys
                and r['layer'].startswith('relu') and float(r['bytes_moved'])>=64*2**20]
    if not references:raise ValueError('No calibration-only bandwidth reference rows')
    rates=np.array([float(r['bytes_moved'])/float(r['gpu_seconds']) for r in references])
    w=float(np.exp(np.mean(np.log(rates))))
    s=np.array([r['S'] for r in training]); b=np.array([r['B'] for r in training])
    y=np.array([r['latency'] for r in training]); e=np.array([r['energy'] for r in training])
    f,d=costs(s,b)
    def residual(log_parameters):
        tau,c=np.exp(log_parameters)
        return np.log(np.maximum(tau,np.maximum(f/c,d/w)).sum(-1)/y)
    rng=np.random.default_rng(seed);fits=[]
    low=np.log([1e-8,1e8]); high=np.log([.1,1e17])
    for i in range(starts):
        initial=np.log([np.percentile(y,10)/17,1e13])+(rng.uniform(-2,2,2) if i else 0)
        fit=least_squares(residual,np.clip(initial,low+1e-6,high-1e-6),bounds=(low,high),max_nfev=3000,
                          ftol=1e-11,xtol=1e-11,gtol=1e-11)
        if fit.success:fits.append((float(np.mean(fit.fun**2)),fit.x))
    if not fits:raise RuntimeError('No converged fits')
    fits.sort(key=lambda x:x[0]);loss,best=fits[0]
    tau,c=map(float,np.exp(best)); theta={'tau':tau,'C':c,'W':w}
    t=latency(s,b,theta);u=np.maximum(0,t-17*tau)
    def energy_residual(log_power):
        p0,p1=np.exp(log_power);return np.log((p0*t+p1*u)/e)
    efits=[]
    for initial in ([100,250],[200,150],[300,20]):
        fit=least_squares(energy_residual,np.log(initial),bounds=(np.log([1e-6,1e-6]),np.log([2000,2000])))
        if fit.success:efits.append((float(np.mean(fit.fun**2)),fit.x))
    if not efits:raise RuntimeError('Energy fit failed')
    eloss,ebest=min(efits,key=lambda x:x[0]);p0,p1=map(float,np.exp(ebest))
    near=np.array([np.exp(x) for objective,x in fits if objective<=loss*1.01+1e-12])
    branches=np.argmax(np.stack([np.full_like(f,tau),f/c,d/w],axis=-1),axis=-1)
    return dict(model='layerwise_roofline',latency=theta,energy={**theta,'p0':p0,'p1':p1},
      units={'tau':'s/operator','C':'FLOPs/s','W':'bytes/s','p0':'W','p1':'W'},
      fit=dict(training_points=len(training),seed=seed,starts=starts,converged_starts=len(fits),
          objective='mean squared log(predicted/measured)',latency_training_log_mse=loss,
          energy_training_log_mse=eloss,energy_method='fit p0,p1 with latency parameters fixed; E=p0*T+p1*(T-17*tau)',
          bandwidth_method='geometric mean logical bytes / CUDA graph device seconds on calibration-only ReLU rows with logical bytes>=64 MiB',
          bandwidth_reference_count=len(references),bandwidth_p10=float(np.percentile(rates,10)),bandwidth_p90=float(np.percentile(rates,90)),
          bandwidth_data_sha256=hashlib.sha256(Path(layer_path).read_bytes()).hexdigest()),
      identifiability=dict(near_optimal_starts=len(near),
          parameter_ranges={name:dict(min=float(near[:,i].min()),max=float(near[:,i].max())) for i,name in enumerate(('tau','C'))},
          active_training_operator_branches={name:int(np.sum(branches==i)) for i,name in enumerate(('launch','compute','memory'))},
          warning='Per-operator roofline assignments are model explanations; empirical throughput and CUDA traces support interpretation, but SM/DRAM counters are permission-blocked. W independently anchored to memory-only operators.'),
      selection='Layerwise form motivated by operator diagnostics; original 69 validation rows have been inspected during development. Fresh proportional 16-point confirmation is excluded from all fitting.')


def error_metrics(measured, predicted):
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    relative = np.abs(predicted/measured-1)
    return dict(n=len(measured), mape_percent=float(100*relative.mean()),
                median_ape_percent=float(100*np.median(relative)),
                p90_ape_percent=float(100*np.percentile(relative, 90)),
                max_ape_percent=float(100*relative.max()),
                log_rmse=float(np.sqrt(np.mean(np.log(predicted/measured)**2))))


def evaluate(rows, parameters):
    predicted_rows = []
    for row in rows:
        s,b = row['S'],row['B']
        item = dict(row)
        item.update(flops=flops(s,b), bytes_moved=bytes_moved(s,b),
                    latency_predicted=latency(s,b,parameters['latency']),
                    memory_predicted=memory(s,b),
                    energy_predicted=energy(s,b,parameters['energy']))
        t = parameters['latency']
        f,d=costs(s,b)
        parts=np.stack([np.full_like(f,t['tau']),f/t['C'],d/t['W']],axis=-1)
        chosen=np.argmax(parts,axis=-1); durations=parts.max(-1)
        totals=[float(durations[chosen==i].sum()) for i in range(3)]
        item['model_branch']=('launch','compute','memory')[int(np.argmax(totals))]
        item.update({name+'_seconds':totals[i] for i,name in enumerate(('launch','compute','memory'))})
        predicted_rows.append(item)
    metrics = {}
    for label, validation in (('calibration',False), ('validation',True)):
        subset = [r for r in predicted_rows if r['is_validation']==validation]
        if not subset:
            raise ValueError(f'No {label} rows')
        metrics[label] = {name:error_metrics([r[name] for r in subset], [r[name+'_predicted'] for r in subset])
                          for name in ('latency','memory','energy')}
    return predicted_rows, metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args()
    path=args.results/'measurements.csv'
    rows=read_measurements(path)
    parameters=fit_parameters([r for r in rows if not r['is_validation']],layer_path=args.results/'layers'/'measurements.csv')
    parameters['measurements_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    parameters['source_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ('calibrate.py','equations.py')}
    predictions,metrics=evaluate(rows,parameters)
    (args.results/'theta.json').write_text(json.dumps(parameters,indent=2)+'\n')
    (args.results/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    with (args.results/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    print(json.dumps(parameters,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
