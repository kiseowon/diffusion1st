"""MLP factorial input contribution: target conditions, Morgan and in-vitro branch."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import repeated_latent_validation as core
import repeat_time_structure_pilot as repeat

LABELS={('absent','basic'):'A_conditions',('present','basic'):'B_conditions_vitro',('absent','morgan'):'C_conditions_morgan',('present','morgan'):'D_conditions_vitro_morgan'}


def arguments(a, mode):
    return ['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(a.smiles),'--out',str(a.out/mode),
        '--source-input',mode,'--input-space','pca','--input-components','64','--output-space','full',
        '--structure-layout','separate','--arms','basic','morgan','--families','mlp',
        '--folds',str(a.folds),'--split-seed',str(a.split_seed),'--model-seeds',*map(str,a.model_seeds),
        '--epochs',str(a.epochs),'--threads',str(a.threads),'--budget-minutes',str(a.budget_minutes),
        '--unit-budget-minutes',str(a.unit_budget_minutes)]


def summarize(a):
    all_reg=[];all_cls=[];dimensions=[]
    for fold in range(a.folds):
        for seed in a.model_seeds:
            left=a.out/'absent'/f'fold_{fold}'/f'seed_{seed}';right=a.out/'present'/f'fold_{fold}'/f'seed_{seed}'
            if not pd.read_csv(left/'compound_split.csv').equals(pd.read_csv(right/'compound_split.csv')):raise ValueError('Compound split differs')
            for folder in [left,right]:
                if not (folder/'completed.json').exists():raise ValueError('Incomplete unit')
            with np.load(left/'baseline_predictions.npz') as x,np.load(right/'baseline_predictions.npz') as y:
                for key in ['target_ids','time_level']:
                    if not np.array_equal(x[key],y[key]):raise ValueError('Targets or baselines differ')
            sizes=[]
            for mode,folder in [('absent',left),('present',right)]:
                for arm in ['basic','morgan']:
                    dim=json.loads((folder/arm/'model_dimensions.json').read_text())
                    for stage,value in dim.items():
                        dimensions.append(dict(condition=LABELS[(mode,arm)],fold=fold,model_seed=seed,stage=stage,**value))
                        sizes.append((stage,value['input_dimension'],value['output_dimension'],value['mlp_parameters']))
                    task=folder/arm/'mlp';record=json.loads((task/'complete.json').read_text())
                    with np.load(task/record['prediction_path']) as p:
                        if not np.array_equal(p['target_ids'],xids(left)):raise ValueError('Prediction target mismatch')
            for stage in ['inner','final']:
                if len({v[1:] for v in sizes if v[0]==stage})!=1:raise ValueError('Nominal model sizes differ across four conditions')
    for mode in ['absent','present']:
        for filename,store in [('pooled_by_model_seed.csv',all_reg),('classification_by_model_seed.csv',all_cls)]:
            table=pd.read_csv(a.out/mode/filename)
            table['condition']=[('baseline_'+model if arm=='baseline' else LABELS[(mode,arm)]) for arm,model in zip(table.arm,table.model)]
            if mode=='present':table=table[table.arm.ne('baseline')]
            store.append(table)
    reg=pd.concat(all_reg,ignore_index=True);cls=pd.concat(all_cls,ignore_index=True)
    if reg.duplicated(['condition','model_seed']).any():raise ValueError('Duplicate pooled metrics')
    reg.to_csv(a.out/'pooled_by_model_seed.csv',index=False);cls.to_csv(a.out/'classification_by_model_seed.csv',index=False)
    metrics=['rmse','mae','pearson_target_mean','spearman_target_mean','r2_probe_variance_weighted']
    summary=reg.groupby('condition')[metrics].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(a.out/'summary.csv')
    classification=cls.groupby(['condition','threshold'])[['macro_f1','down_f1','up_f1','active_sign_accuracy','actual_active_fraction','predicted_active_fraction']].agg(['mean','std'])
    classification.columns=['_'.join(c) for c in classification.columns];classification.to_csv(a.out/'classification_summary.csv')
    wide=reg[reg.model_seed.ge(0)].pivot(index='model_seed',columns='condition',values='rmse');paired=[]
    for seed,row in wide.iterrows():
        for name,left,right in [('vitro_without_structure_B_minus_A','B_conditions_vitro','A_conditions'),('vitro_with_structure_D_minus_C','D_conditions_vitro_morgan','C_conditions_morgan'),('structure_without_vitro_C_minus_A','C_conditions_morgan','A_conditions'),('structure_with_vitro_D_minus_B','D_conditions_vitro_morgan','B_conditions_vitro')]:
            paired.append(dict(model_seed=int(seed),comparison=name,rmse_difference=row[left]-row[right]))
    pd.DataFrame(paired).to_csv(a.out/'paired_rmse.csv',index=False)
    pd.DataFrame(dimensions).to_csv(a.out/'model_dimensions.csv',index=False)
    print('\nFOUR INPUT CONDITIONS\n'+summary.to_string(),flush=True)
    print('\nPAIRED DIFFERENCES (negative is lower error)\n'+pd.DataFrame(paired).to_string(index=False),flush=True)


def xids(folder):
    with np.load(folder/'baseline_predictions.npz') as saved:return saved['target_ids'].copy()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','structure','smiles','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--folds',type=int,default=5);p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--model-seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--epochs',type=int,default=100);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--budget-minutes',type=float,default=1440);p.add_argument('--unit-budget-minutes',type=float,default=1440);p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if a.folds<2 or min(a.epochs,a.threads,a.budget_minutes,a.unit_budget_minutes)<=0:p.error('Invalid counts/budgets')
    if len(a.model_seeds)!=len(set(a.model_seeds)) or any(s<0 or s>=2**32 for s in [a.split_seed]+a.model_seeds):p.error('Invalid seeds')
    jobs=4*a.folds*len(a.model_seeds)
    print(json.dumps(dict(mode='execute' if a.execute else 'plan_only',jobs=jobs,fits_up_to=2*jobs,family='mlp',conditions=list(LABELS.values()),input_pca=64,output='full_probes',source_ablation='expression + source metadata + observation masks together, not expression alone',limitations='Same nominal architecture, different effective inputs. Fixed folds reused for initialization repeats; exploratory, no p-values or confirmatory testing.'),indent=2),flush=True)
    if not a.execute:return jobs
    core.protect_output(a.out,[a.prepared,a.structure,a.smiles])
    sources=list(Path(__file__).parent.glob('*.py'))+[core.SNAPSHOT]
    core.contract_folder(a.out,dict(arguments={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='execute'},source_sha256={str(p.resolve()):core.digest(p) for p in sources}))
    for mode in ['absent','present']:
        repeat.main(arguments(a,mode)+['--execute'])
        if not json.loads((a.out/mode/'repeat_status.json').read_text())['all_complete']:
            core.save_json(a.out/'information_status.json',dict(all_complete=False,partial_source_mode=mode));return jobs
    summarize(a)
    core.save_json(a.out/'information_status.json',dict(all_complete=True,jobs=jobs,independent_split_repeats=False,initialization_repeats=True,source_branch_is_expression_plus_metadata=True))
    return jobs

if __name__=='__main__':main()
