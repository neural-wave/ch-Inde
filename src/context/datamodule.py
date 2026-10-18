"""Calendar-exact, same-zone context; training never reads held-out labels."""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datamodule import DailyPriceDataModule, ZONES, validate_training_frame


class ContextDataset(Dataset):
    def __init__(self, base, history, lag_days=1):
        self.base, self.frame = base, base.frame
        self.history, self.lag_days = history, lag_days
        self.context_dates = []
        self.context_values = []
        for i in range(len(base)):
            row = base[i]
            previous = (date.fromisoformat(row['delivery_date']) - timedelta(days=lag_days)).isoformat()
            value = history.get((previous, int(row['zone'])))
            self.context_dates.append(previous if value is not None else None)
            self.context_values.append(value)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = dict(self.base[index])
        value = self.context_values[index]
        row['history'] = torch.zeros(96) if value is None else value.clone()
        row['history_available'] = torch.tensor(value is not None)
        # Empty string is collatable and records absence without invented dates.
        row['history_date'] = self.context_dates[index] or ''
        return row


def history_lookup(frame):
    return {(day, ZONES.index(zone)): torch.tensor(
        group.sort_values(['hour_index', 'quarter_index']).price_eur_mwh.to_numpy(dtype='float32'))
        for (day, zone), group in frame.groupby(['delivery_date', 'market_id'], sort=True)}


class ContextDataModule(DailyPriceDataModule):
    def __init__(self, train_path: str = 'data/train_prices_15min.csv', predict_path: str | None = None,
                 batch_size: int = 32, num_workers: int = 0, train_end: str = '2026-06-30',
                 val_start: str = '2026-07-01', val_end: str = '2026-07-31',
                 test_start: str = '2026-08-01', test_end: str = '2026-09-30',
                 split_protocol: str = 'static', cv_fold: int = 0,
                 training_window_stride_hours: int | None = None, split_manifest: str | None = None,
                 history_path: str | None = None, lag_days: int = 1):
        if training_window_stride_hours is not None:
            raise ValueError('Context experiments use fixed calendar days, not sliding windows')
        if lag_days not in (1, 7):
            raise ValueError('Context lag must be 1 or 7 calendar days')
        super().__init__(train_path, predict_path, batch_size, num_workers, train_end, val_start,
                         val_end, test_start, test_end, split_protocol, cv_fold,
                         training_window_stride_hours, split_manifest)
        self.history_path, self.lag_days = history_path, lag_days
        self.history_inventory = {}

    def setup(self, stage=None):
        super().setup(stage)
        if stage == 'predict':
            # Jury inference needs only hourly input, supplied history, checkpoint statistics.
            history = (history_lookup(validate_training_frame(pd.read_csv(self.history_path)))
                       if self.history_path else {})
            self._sets['predict'] = ContextDataset(self._sets['predict'], history, self.lag_days)
            return
        frame = validate_training_frame(pd.read_csv(self.train_path))
        train_dates = set(self.split_dates['train'])
        training_history = history_lookup(frame[frame.delivery_date.isin(train_dates)])
        # Rolling-realistic validation may use earlier validation outcomes. Reserved,
        # unused, and test outcomes are excluded; exact calendar lookup enforces D-1.
        validation_dates = train_dates | set(self.split_dates['val'])
        validation_history = history_lookup(frame[frame.delivery_date.isin(validation_dates)])
        test_dates = validation_dates | set(self.split_dates['test'])
        test_history = history_lookup(frame[frame.delivery_date.isin(test_dates)])
        for name, history in [('train', training_history), ('val', validation_history), ('test', test_history)]:
            dataset = ContextDataset(self._sets[name], history, self.lag_days)
            self._sets[name] = dataset
            self.history_inventory[name] = [
                {'delivery_date': dataset.base[i]['delivery_date'], 'zone': int(dataset.base[i]['zone']),
                 'history_date': day, 'history_available': day is not None}
                for i, day in enumerate(dataset.context_dates)]
