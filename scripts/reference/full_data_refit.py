"""Refit the frozen HARVEST winner on every supplied date and export its bundle."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.agent.context_rope_experiments import model_configuration


def main():
    import pandas as pd
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'config/full_data_harvest_winner.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config['epochs']!=30 or config['seed']!=42 or config['weights']!=[.5,.5]:
        raise ValueError('Freeze the selected seed-42, 30-epoch, 50/50 recipe')
    out = args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    dates = sorted(pd.read_csv(ROOT/'data/train_prices_15min.csv').delivery_date.unique())
    split = dict(strategy='full_data_refit',train=dates,validation=[],test=[],reserved=[],unused=[])
    split_path = out/'split.json';split_path.write_text(json.dumps(split,indent=2))
    source = json.loads(args.source_manifest.read_text())
    result = dict(configuration=config,source_manifest=source,training_dates=dates,
                  interpretation='Full-data deployment refit; all scores are in-sample diagnostics',runs=[])
    bundle = out/'bundle';bundle.mkdir()
    recipe = dict(format_version=1,seed=42,trained_epochs=30,weights=config['weights'],
                  training_date_count=len(dates),training_date_start=dates[0],training_date_end=dates[-1],
                  source_sha256=source['source_sha256'],git_revision=source['git_revision'],
                  reference_harvest_mae=config['reference_harvest_mae'],
                  reference_interpretation='Earlier held-out HARVEST score; not a score of this full-data refit',components=[])
    for candidate in config['components']:
        rid=f'{candidate["family"].lower()}-{candidate["label"].lower()}-s42-e30'
        settings=model_configuration(config,candidate)
        settings['data']['split_manifest']=str(split_path)
        settings['trainer'].update(limit_val_batches=0,num_sanity_val_steps=0)
        settings['run'].update(run_id=rid,selection_metric='final_epoch')
        resolved=out/(rid+'.yaml');resolved.write_text(yaml.safe_dump(settings))
        logging.info('full_data_fit family=%s label=%s dates=%s epochs=30',candidate['family'],candidate['label'],len(dates))
        with (out/(rid+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-m','src.context.full_refit_cli','fit','--config',str(resolved)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        leaf=ROOT/'artifacts/runs'/config['experiment']/rid
        record=json.loads((leaf/'final_scores.json').read_text())
        if record['trained_epochs']!=30 or record['training_dates']!=len(dates):
            raise RuntimeError('Full-data budget or coverage differs')
        record.update(**candidate,candidate=candidate['family']+'-base/'+candidate['label'],seed=42,stage='train')
        result['runs'].append(record)
        folder=bundle/rid;folder.mkdir()
        files=['config.yaml','run.json','final_scores.json','history_inventory.json.gz','checkpoints/last.ckpt']
        suffix='_gated' if 'train_calibrated_gate' in record['metrics'] else ''
        files += [f'train{suffix}_predictions.csv']
        if (leaf/'gate.json').exists():files.append('gate.json')
        for name in files:
            target=folder/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(leaf/name,target)
        component=dict(label=record['candidate'],weight=.5,checkpoint=f'{rid}/checkpoints/last.ckpt',
                       checkpoint_sha256=record['checkpoint_sha256'],config=f'{rid}/config.yaml',
                       train_predictions=f'{rid}/train{suffix}_predictions.csv')
        if (folder/'gate.json').exists():component['gate']=f'{rid}/gate.json'
        recipe['components'].append(component)
        (out/'results.json').write_text(json.dumps(result,indent=2))
    shutil.copy2(split_path,bundle/'split.json')
    shutil.copy2(args.source_manifest,bundle/'source_manifest.json')
    (bundle/'ensemble.json').write_text(json.dumps(recipe,indent=2))
    with tarfile.open(out/'bundle.tar.gz','w:gz') as archive:archive.add(bundle,arcname='bundle')
    (out/'COMPLETE').write_text('Both full-data components completed exactly 30 epochs; deployment bundle ready.\n')
    logging.info('full_data_refit_complete components=2 dates=%s bundle=%s',len(dates),bundle)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
