"""Explicit deployment-only train-all partition; normal evaluation rules stay strict."""
import json
from pathlib import Path

import pandas as pd

from src.datamodule import validate_training_frame
from .retrieval_datamodule import RetrievalDataModule


def full_assignments(frame, split):
    dates = sorted(frame.delivery_date.unique())
    train = split.get('train')
    if not isinstance(train,list) or any(not isinstance(day,str) for day in train):
        raise ValueError('Full-data training dates must be a list of date strings')
    if sorted(train) != dates or len(train) != len(dates):
        raise ValueError('Full-data refit must use every supplied date exactly once')
    if any(split.get(key) for key in ('validation','test','reserved','unused')):
        raise ValueError('Full-data refit has no holdout or reserved dates')
    return dict(train=train,validation=[],test=[],reserved=[],unused=[])


def validate_full_split(train_path,split_path):
    frame=validate_training_frame(pd.read_csv(train_path))
    return sorted(full_assignments(frame,json.loads(Path(split_path).read_text()))['train'])


class FullRefitDataModule(RetrievalDataModule):
    def _read_split_manifest(self,frame):
        split=json.loads(Path(self.split_manifest).read_text())
        if split.get('strategy')!='full_data_refit':
            raise ValueError('Use an explicitly labeled full_data_refit manifest')
        return full_assignments(frame,split)
