"""Causal same-zone retrieval using known hourly curves and past corrections."""
from datetime import date

import numpy as np

from src.classical.structured import makima_centered

POLICIES = {'R0': ('yesterday', 1), 'R1': ('shape_regime', 28),
            'R2': ('gradient', 28), 'R3': ('local', 28), 'R4': ('local', 90)}


def local_slopes(hourly):
    delta = np.diff(hourly, axis=-1)
    return np.concatenate((delta[..., :1], delta), -1), np.concatenate((delta, delta[..., -1:]), -1)


def distances(current, previous, metric):
    if metric == 'local':
        left, right = local_slopes(current)
        old_left, old_right = local_slopes(previous)
        squared = .5*((left-old_left)**2+(right-old_right)**2)/25**2
        return np.sqrt(squared + .2*((current-previous)/60)**2).T
    if metric == 'gradient':
        value = np.mean(((np.diff(current)-np.diff(previous, axis=-1))/25)**2, -1)
    else:
        mean, std = current.mean(), current.std()
        means, stds = previous.mean(-1), previous.std(-1)
        shape = (current-mean)/max(std, 1.)
        shapes = (previous-means[:, None])/np.maximum(stds[:, None], 1.)
        value = np.mean((shape-shapes)**2, -1)
        if metric == 'shape_regime':
            value += .2*((mean-means)/60)**2 + .2*((std-stds)/25)**2
        elif metric != 'shape':
            raise ValueError('Unknown retrieval distance')
    return np.broadcast_to(np.sqrt(value)[None], (24, len(previous)))


def inverse_distance_weights(distance):
    zeros = distance <= 1e-12
    if zeros.any():
        return zeros.astype(float)/zeros.sum()
    weights = 1/np.maximum(distance, 1e-8)
    return weights/weights.sum()


class RetrievalIndex:
    def __init__(self, history):
        self.zones = {}
        for zone in range(4):
            keys = sorted(day for day, z in history if z == zone)
            values = np.stack([np.asarray(history[(day, zone)], dtype=float) for day in keys]) if keys else np.empty((0, 96))
            hourly = values.reshape(-1, 24, 4).mean(-1)
            correction = values.reshape(-1, 24, 4)-hourly[..., None]-2*makima_centered(hourly)
            correction -= correction.mean(-1, keepdims=True)
            self.zones[zone] = (keys, np.array([date.fromisoformat(day).toordinal() for day in keys]), hourly, correction)

    def retrieve(self, day, zone, hourly, policy):
        metric, lookback = POLICIES[policy]
        keys, ordinals, profiles, corrections = self.zones[zone]
        current = date.fromisoformat(day).toordinal()
        valid = (ordinals < current) & (ordinals >= current-lookback)
        if metric == 'yesterday':
            valid &= ordinals == current-1
        pool = np.flatnonzero(valid)
        signal, confidence = np.zeros((24, 4)), np.zeros((24, 3))
        inventory = {'delivery_date': day, 'zone': zone, 'policy': policy, 'candidate_days': len(pool),
                     'history_available': bool(len(pool)), 'candidate_dates': [keys[i] for i in pool],
                     'indices': [], 'weights': [], 'distances': []}
        if not len(pool):
            return signal.reshape(96), confidence, inventory
        distance = distances(np.asarray(hourly, dtype=float), profiles[pool], 'local' if metric == 'yesterday' else metric)
        for hour in range(24):
            # Stable ties prefer the most recent eligible date.
            selected = np.lexsort((-ordinals[pool], distance[hour]))[:min(5, len(pool))]
            d = distance[hour, selected]
            weights = inverse_distance_weights(d)
            values = corrections[pool[selected], hour]
            signal[hour] = weights @ values
            disagreement = np.sqrt(np.sum(weights[:, None]*(values-signal[hour])**2)/4)
            confidence[hour] = (weights @ d, disagreement, len(selected)/5)
            inventory['indices'].append(selected.tolist())
            inventory['weights'].append(weights.tolist())
            inventory['distances'].append(d.tolist())
        signal -= signal.mean(-1, keepdims=True)
        return signal.reshape(96), confidence, inventory
