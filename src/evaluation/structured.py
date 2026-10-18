"""Auditable, serialized evaluation of whole-date reconstruction experiments."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.constraints import write_predictions


def dataset_arrays(dataset):
    rows = [dataset[i] for i in range(len(dataset))]
    return {
        'hourly': np.stack([r['hourly_original'].numpy() for r in rows]),
        'target': dataset.frame.sort_values(['delivery_date', 'market_id', 'hour_index', 'quarter_index']).price_eur_mwh.to_numpy(dtype=float).reshape(-1, 96),
        'zones': np.array([int(r['zone']) for r in rows]),
        'dates': [r['delivery_date'] for r in rows],
    }


def evaluate_predictions(prediction, data, path, reference=None):
    """Score the actual CSV, preserving daily/equal-zone aggregation."""
    write_predictions(prediction, data['hourly'], data['dates'], data['zones'], path)
    frame = pd.read_csv(path)
    prediction = frame.predicted_price_eur_mwh.to_numpy().reshape(-1, 24, 4)
    target = data['target'].reshape(-1, 24, 4).astype(float)
    error = np.abs(prediction-target)
    zones = data['zones']
    per_zone = {f'ZONE_{z}': float(error[zones == z].mean()) for z in range(4)}
    flat = np.abs(target-data['hourly'][..., None])
    metrics = {
        'mae': float(np.mean(list(per_zone.values()))),
        'flat_mae': float(np.mean([flat[zones == z].mean() for z in range(4)])),
        'max_hourly_mean_error': float(np.abs(prediction.mean(-1)-data['hourly']).max()),
        'per_zone': per_zone,
        'per_hour': error.mean((0, 2)).tolist(),
        'per_quarter': error.mean((0, 1)).tolist(),
        'days': len(set(data['dates'])),
    }
    basis = np.array([[-3/np.sqrt(20), .5, -1/np.sqrt(20)],
                      [-1/np.sqrt(20), -.5, 3/np.sqrt(20)],
                      [1/np.sqrt(20), -.5, -3/np.sqrt(20)],
                      [3/np.sqrt(20), .5, 1/np.sqrt(20)]])
    predicted_residual = prediction-data['hourly'][..., None]
    true_residual = target-data['hourly'][..., None]
    pc, tc = predicted_residual@basis, true_residual@basis
    metrics['coefficient_mae'] = {name: float(np.abs(pc[..., i]-tc[..., i]).mean())
                                for i, name in enumerate(['slope', 'curvature', 'asymmetry'])}
    metrics['slope_sign_accuracy'] = float((np.sign(pc[..., 0]) == np.sign(tc[..., 0])).mean())
    true_amplitude, predicted_amplitude = np.abs(true_residual).mean(-1), np.abs(predicted_residual).mean(-1)
    active = true_amplitude > 1
    metrics['false_flat_rate_active_hours'] = float((predicted_amplitude[active] < .1).mean()) if active.any() else None
    metrics['overshoot_rate'] = float((predicted_amplitude > 2*true_amplitude+1).mean())
    for name, mask in {
        'near_zero': np.abs(data['hourly']) <= 1,
        'negative': data['hourly'] < 0,
        'steep': np.maximum(np.abs(np.diff(data['hourly'], prepend=data['hourly'][:, :1])),
                            np.abs(np.diff(data['hourly'], append=data['hourly'][:, -1:]))) > 30,
    }.items():
        metrics[name] = {'hours': int(mask.sum()), 'mae': float(error[mask].mean()) if mask.any() else None}
    if reference is not None:
        metrics['makima_mae'] = reference
        metrics['improvement_over_makima'] = reference-metrics['mae']
        metrics['improvement_percent'] = 100*(reference-metrics['mae'])/reference
    daily = pd.DataFrame({'delivery_date': data['dates'], 'market_id': [f'ZONE_{z}' for z in zones],
                          'mae': error.mean((1, 2))})
    daily.to_csv(Path(path).with_name(Path(path).stem+'_daily.csv'), index=False)
    return metrics


def flatten_metrics(metrics, prefix):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out.update(flatten_metrics(value, prefix+'/'+key))
        elif isinstance(value, (float, int)) and value is not None:
            out[prefix+'/'+key] = value
    return out
