"""Fold-aware rolling retrieval; no held-out or future labels in training."""
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datamodule import DailyPriceDataModule, validate_training_frame, ZONES
from .datamodule import ContextDataModule
from .retrieval import POLICIES, RetrievalIndex


class RetrievalDataset(Dataset):
    def __init__(self, base, index, policy):
        self.base, self.frame = base, base.frame
        self.context_values, self.confidences, self.retrieval_inventory = [], [], []
        for i in range(len(base)):
            row = base[i]
            hourly = row.get('hourly_original', row['hourly']).numpy()
            signal, confidence, inventory = index.retrieve(row['delivery_date'], int(row['zone']), hourly, policy)
            self.context_values.append(torch.tensor(signal, dtype=torch.float32))
            self.confidences.append(torch.tensor(confidence, dtype=torch.float32))
            self.retrieval_inventory.append(inventory)
        self.context_dates = [entry['candidate_dates'] if entry['history_available'] else None
                              for entry in self.retrieval_inventory]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = dict(self.base[index])
        row['history'] = self.context_values[index].clone()
        row['history_confidence'] = self.confidences[index].clone()
        row['history_available'] = torch.tensor(self.retrieval_inventory[index]['history_available'])
        return row


class RetrievalDataModule(ContextDataModule):
    dataset_class=RetrievalDataset
    def __init__(self, train_path: str = 'data/train_prices_15min.csv', predict_path: str | None = None,
                 batch_size: int = 32, num_workers: int = 0, train_end: str = '2026-06-30',
                 val_start: str = '2026-07-01', val_end: str = '2026-07-31',
                 test_start: str = '2026-08-01', test_end: str = '2026-09-30',
                 split_protocol: str = 'static', cv_fold: int = 0,
                 training_window_stride_hours: int | None = None, split_manifest: str | None = None,
                 history_path: str | None = None, lag_days: int = 1, retrieval_policy: str = 'R3'):
        super().__init__(train_path, predict_path, batch_size, num_workers, train_end, val_start,
                         val_end, test_start, test_end, split_protocol, cv_fold,
                         training_window_stride_hours, split_manifest, history_path, lag_days)
        if retrieval_policy not in POLICIES:
            raise ValueError('Expected R0–R4 retrieval policy')
        self.retrieval_policy = retrieval_policy

    def setup(self, stage=None):
        DailyPriceDataModule.setup(self, stage)
        if stage == 'predict':
            frame = validate_training_frame(pd.read_csv(self.history_path)) if self.history_path else None
            history = retrieval_history_lookup(frame) if frame is not None else {}
            self._sets['predict'] = self.dataset_class(self._sets['predict'], RetrievalIndex(history), self.retrieval_policy)
            return
        frame = validate_training_frame(pd.read_csv(self.train_path))
        allowed = set(self.split_dates['train'])
        self.history_inventory = {}
        for name in ('train', 'val', 'test'):
            if name != 'train':
                allowed |= set(self.split_dates[name])
            history = retrieval_history_lookup(frame[frame.delivery_date.isin(allowed)])
            dataset = self.dataset_class(self._sets[name], RetrievalIndex(history), self.retrieval_policy)
            self._sets[name] = dataset
            self.history_inventory[name] = dataset.retrieval_inventory


def retrieval_history_lookup(frame):
    # Rank neighbors in source precision; convert only neural inputs to float32.
    return {(day, ZONES.index(zone)): group.sort_values(['hour_index','quarter_index']).price_eur_mwh.to_numpy(dtype='float64')
            for (day,zone),group in frame.groupby(['delivery_date','market_id'],sort=True)}
