"""Paired input-pipeline comparison: PCA64 versus full probes, full-probe output."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import repeat_time_structure_pilot as repeat
import repeated_latent_validation as core


def arguments(a, mode):
    return ['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(a.smiles),
        '--out',str(a.out/mode),'--input-space',mode,'--input-components','64','--output-space','full',
        '--structure-layout','separate','--arms',*a.arms,'--families',*a.families,
        '--folds',str(a.folds),'--split-seed',str(a.split_seed),'--model-seeds',*map(str,a.model_seeds),
        '--epochs',str(a.epochs),'--threads',str(a.threads),'--budget-minutes',str(a.budget_minutes),
        '--unit-budget-minutes',str(a.unit_budget_minutes),'--cache-diffusion-context']


def compare(a):
    summaries=[];records=[];dimensions=[]
    for fold in range(a.folds):
        for seed in a.model_seeds:
            left=a.out/'pca'/f'fold_{fold}'/f'seed_{seed}'
            right=a.out/'full'/f'fold_{fold}'/f'seed_{seed}'
            for path in [left,right]:
                if not (path/'completed.json').exists():raise ValueError('Incomplete unit: '+str(path))
            if not pd.read_csv(left/'compound_split.csv').equals(pd.read_csv(right/'compound_split.csv')):
                raise ValueError('Compound split mismatch')
            with np.load(left/'baseline_predictions.npz') as x,np.load(right/'baseline_predictions.npz') as y:
                for key in ['target_ids','time_level']:
                    if not np.array_equal(x[key],y[key]):raise ValueError('Baseline/target mismatch')
            for arm in a.arms:
                for mode,folder in [('pca',left),('full',right)]:
                    dim=json.loads((folder/arm/'model_dimensions.json').read_text())
                    for stage,value in dim.items():dimensions.append(dict(input_space=mode,fold=fold,model_seed=seed,arm=arm,stage=stage,**value))
                for family in a.families:
                    predictions=[]
                    for folder in [left,right]:
                        task=folder/arm/family;marker=json.loads((task/'complete.json').read_text())
                        with np.load(task/marker['prediction_path']) as saved:predictions.append(saved['target_ids'].copy())
                    if not np.array_equal(*predictions):raise ValueError('Model prediction targets mismatch')
    for mode in ['pca','full']:
        table=pd.read_csv(a.out/mode/'pooled_by_model_seed.csv')
        summaries.append(table.assign(input_space=mode))
    all_metrics=pd.concat(summaries,ignore_index=True)
    all_metrics.to_csv(a.out/'pooled_metrics.csv',index=False)
    model=all_metrics[all_metrics.model_seed.ge(0)]
    index=['model_seed','arm','model']
    if model.duplicated(index+['input_space']).any():raise ValueError('Duplicate pooled metrics')
    for keys,group in model.groupby(index):
        if set(group.input_space)!=set(['pca','full']):raise ValueError('Unpaired modes')
        p=group.set_index('input_space')
        records.append(dict(zip(index,keys),pca_rmse=p.loc['pca','rmse'],full_rmse=p.loc['full','rmse'],full_minus_pca_rmse=p.loc['full','rmse']-p.loc['pca','rmse'],pca_mae=p.loc['pca','mae'],full_mae=p.loc['full','mae'],pca_pearson=p.loc['pca','pearson_target_mean'],full_pearson=p.loc['full','pearson_target_mean']))
    paired=pd.DataFrame(records);paired.to_csv(a.out/'paired_by_model_seed.csv',index=False)
    summary=paired.groupby(['arm','model']).agg(pca_rmse_mean=('pca_rmse','mean'),pca_rmse_std=('pca_rmse','std'),full_rmse_mean=('full_rmse','mean'),full_rmse_std=('full_rmse','std'),full_minus_pca_mean=('full_minus_pca_rmse','mean'),full_minus_pca_std=('full_minus_pca_rmse','std'),pca_mae_mean=('pca_mae','mean'),full_mae_mean=('full_mae','mean'),pca_pearson_mean=('pca_pearson','mean'),full_pearson_mean=('full_pearson','mean'))
    summary.to_csv(a.out/'comparison_summary.csv')
    pd.DataFrame(dimensions).to_csv(a.out/'model_dimensions.csv',index=False)
    for filename in ['classification_summary.csv','classification_by_model_seed.csv']:
        paths=[a.out/mode/filename for mode in ['pca','full']]
        if all(p.exists() for p in paths):pd.concat([pd.read_csv(p).assign(input_space=mode) for p,mode in zip(paths,['pca','full'])],ignore_index=True).to_csv(a.out/('combined_'+filename),index=False)
    print('\nPAIRED INPUT PIPELINE COMPARISON\n'+summary.to_string(),flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','structure','smiles','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--families',nargs='+',choices=['mlp','cgan','ddpm'],default=['mlp'])
    p.add_argument('--arms',nargs='+',choices=['basic','morgan'],default=['basic','morgan'])
    p.add_argument('--folds',type=int,default=5);p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--model-seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--budget-minutes',type=float,default=1440);p.add_argument('--unit-budget-minutes',type=float,default=1440)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    if a.folds<2 or min(a.epochs,a.threads,a.budget_minutes,a.unit_budget_minutes)<=0:p.error('Invalid counts/budgets')
    if any(len(v)!=len(set(v)) for v in [a.arms,a.families,a.model_seeds]):p.error('Duplicate conditions/seeds')
    if any(s<0 or s>=2**32 for s in [a.split_seed]+a.model_seeds):p.error('Invalid seeds')
    jobs=2*a.folds*len(a.model_seeds)*len(a.arms)*len(a.families)
    print(json.dumps(dict(mode='execute' if a.execute else 'plan_only',jobs=jobs,fits_up_to=2*jobs,input_modes=['pca64','full'],output='all_probes',families=a.families,arms=a.arms,folds=a.folds,model_seeds=a.model_seeds,limitations='First-layer parameter count and PCA-score scaling differ. This compares input pipelines, not isolated PCA information loss. Exploratory, no independent confirmatory test or significance testing.'),indent=2),flush=True)
    if not a.execute:return jobs
    core.protect_output(a.out,[a.prepared,a.structure,a.smiles])
    sources=list(Path(__file__).parent.glob('*.py'))+[core.SNAPSHOT]
    core.contract_folder(a.out,dict(arguments={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='execute'},source_sha256={str(f.resolve()):core.digest(f) for f in sources}))
    for mode in ['pca','full']:
        repeat.main(arguments(a,mode)+['--execute'])
        status=json.loads((a.out/mode/'repeat_status.json').read_text())
        if not status['all_complete']:
            core.save_json(a.out/'comparison_status.json',dict(all_complete=False,partial_mode=mode))
            print('PARTIAL: resume with identical command; no complete comparison yet',flush=True)
            return jobs
    compare(a)
    core.save_json(a.out/'comparison_status.json',dict(all_complete=True,jobs=jobs,independent_split_repeats=False,model_seed_repeats=True,uncertainty='initialization variation, not compound sampling CI',selection='all seeds retained; no post-hoc winner selection'))
    return jobs

if __name__=='__main__':main()
