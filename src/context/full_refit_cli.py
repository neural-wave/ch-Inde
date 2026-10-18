"""Fixed-budget full-data training, with training diagnostics and no holdout score."""
import gzip
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import EarlyStopping

from src.classical.structured import makima_centered
from src.evaluation.structured import dataset_arrays, evaluate_predictions, flatten_metrics
from src.cli import DailyTransformerCLI
from src.gating import gate_to_flat, select_gate
from .full_refit_datamodule import FullRefitDataModule, validate_full_split
from .rope_model import RopeContextTransformer


class FullDataDiagnostics(Callback):
    def on_fit_end(self, trainer, pl_module):
        dm = trainer.datamodule
        if dm.split_dates['val'] or dm.split_dates['test']:
            raise RuntimeError('Full-data diagnostics require training dates only')
        root = Path(trainer.checkpoint_callback.dirpath).parent
        checkpoint = root/'checkpoints/last.ckpt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        epochs = int(state['loops']['fit_loop']['epoch_progress']['current']['processed'])
        del state
        if epochs != trainer.max_epochs:
            raise RuntimeError('Full-data checkpoint did not complete its fixed budget')
        model = type(pl_module).load_from_checkpoint(checkpoint, map_location=pl_module.device, weights_only=False).eval()
        data = dataset_arrays(dm._sets['train'])
        with torch.inference_mode():
            prediction = torch.cat([model.batch_prediction({key: value.to(pl_module.device) if torch.is_tensor(value) else value
                                    for key, value in batch.items()}).cpu() for batch in dm._loader('train', False)]).numpy()
        reference = float(np.abs(data['hourly'][..., None]+2*makima_centered(data['hourly'])-data['target'].reshape(-1,24,4)).mean())
        scores = evaluate_predictions(prediction, data, root/'train_predictions.csv', reference)
        if model.tokenization_config['mode'] == 'masked':
            gate = select_gate(prediction, data['target'], data['hourly'])
            gate['training_fit_mae'] = gate.pop('validation_mae')
            gate['ungated_training_mae'] = gate.pop('ungated_validation_mae')
            gate['protocol'] = 'All original full-data training dates; fixed grid; radius=1'
            gate['training_dates_sha256'] = hashlib.sha256(json.dumps(dm.split_dates['train']).encode()).hexdigest()
            (root/'gate.json').write_text(json.dumps(gate, indent=2))
            scores['train_calibrated_gate'] = evaluate_predictions(
                gate_to_flat(prediction, data['hourly'], **gate['params']), data, root/'train_gated_predictions.csv', reference)
        with gzip.open(root/'history_inventory.json.gz','wt') as stream:
            json.dump(dm.history_inventory, stream, separators=(',',':'))
        for entries in dm.history_inventory.values():
            for entry in entries:
                if any(day >= entry['delivery_date'] for day in entry['candidate_dates']):
                    raise RuntimeError('History must strictly precede each current date')
        record = dict(metrics=scores, train_metrics=scores, evaluation_stage='train',
            interpretation='In-sample training diagnostics only; no validation or test score',
            trained_epochs=epochs, evaluated_epoch_count=epochs, selection_metric='final_epoch',
            checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            artifact_path=str(root), training_dates=len(dm.split_dates['train']),
            training_samples=len(data['dates']), parameter_count=sum(p.numel() for p in model.parameters()))
        if trainer.logger:
            trainer.logger.log_metrics(flatten_metrics(scores,'full_data_train_serialized'), step=trainer.global_step)
            suffix = '_gated' if 'train_calibrated_gate' in scores else ''
            trainer.logger.log_table('full_data_training_daily_scores', dataframe=pd.read_csv(root/f'train{suffix}_predictions_daily.csv'))
            record['wandb_url'] = trainer.logger.experiment.url
        (root/'final_scores.json').write_text(json.dumps(record, indent=2))
        metrics_path = root/'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics.update(flatten_metrics(scores,'full_data_train_serialized'))
        metrics_path.write_text(json.dumps(metrics,indent=2))
        metadata_path = root/'run.json'
        metadata = json.loads(metadata_path.read_text())
        metadata.update(full_data_refit=True, evaluation_metric='full_data_train_serialized/mae',
                        evaluation_interpretation=record['interpretation'])
        metadata_path.write_text(json.dumps(metadata,indent=2))


class FullDataCLI(DailyTransformerCLI):
    def before_fit(self):
        cfg = self.config_init.fit
        if not cfg.run.smoke_test and self.trainer.max_epochs != 30:
            raise ValueError('Full-data refits require exactly 30 epochs')
        if cfg.run.selection_metric != 'final_epoch':
            raise ValueError('Full-data refits must use final epoch selection')
        validate_full_split(cfg.data.train_path, cfg.data.split_manifest)
        super().before_fit()
        self.trainer.callbacks = [cb for cb in self.trainer.callbacks if not isinstance(cb, EarlyStopping)]
        self.trainer.callbacks.append(FullDataDiagnostics())


def main():
    FullDataCLI(model_class=RopeContextTransformer, datamodule_class=FullRefitDataModule,
                save_config_callback=None, run=True, parser_kwargs={'default_config_files':[]})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    main()
