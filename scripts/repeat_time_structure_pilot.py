"""Repeat time-preserving structure comparisons; Descriptor excluded by design."""
import argparse
import json
from pathlib import Path
import time
import meeting_time_structure_pilot as pilot
import repeated_latent_validation as core
from evaluate_repeated_predictions import score


def arguments(a,fold,seed):
    return ['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(a.smiles),
        '--out',str(a.out/f'fold_{fold}'/f'seed_{seed}'),'--epochs',str(a.epochs),
        '--threads',str(a.threads),'--budget-minutes',str(a.unit_budget_minutes),
        '--split-seed',str(a.split_seed),'--model-seed',str(seed),'--fold',str(fold),'--folds',str(a.folds),
        '--arms',*a.arms,'--families',*a.families]


def summarize(a):
    import numpy as np
    import pandas as pd
    import hashlib
    complete=[];unit_rows=[];seen_baselines={}
    for fold in range(a.folds):
        for seed in a.model_seeds:
            folder=a.out/f'fold_{fold}'/f'seed_{seed}'
            if (folder/'results.csv').exists():
                frame=pd.read_csv(folder/'results.csv').assign(fold=fold,model_seed=seed)
                unit_rows.append(frame)
            if not (folder/'completed.json').exists():continue
            complete.append((fold,seed))
            with np.load(folder/'baseline_predictions.npz',allow_pickle=False) as saved:
                ids=saved['target_ids'].copy();pred=saved['time_level'].copy()
            fingerprint=hashlib.sha256(ids.tobytes()+pred.tobytes()).hexdigest()
            if fold in seen_baselines and fingerprint!=seen_baselines[fold][0]:raise ValueError('Baseline differs across model seeds')
            seen_baselines[fold]=(fingerprint,ids,pred)
    if unit_rows:pd.concat(unit_rows,ignore_index=True).to_csv(a.out/'by_fold_model_seed.csv',index=False)
    # Only full fold coverage is eligible for a pooled OOF evaluation.
    seeds=[s for s in a.model_seeds if all((f,s) in complete for f in range(a.folds))]
    if not seeds:
        print('No model seed has all folds completed; by-fold partial results only.',flush=True)
        return False
    _,_,all_ids,meta,Y,_,_=core.load_data(a)
    positions={int(cid):i for i,cid in enumerate(all_ids)}
    regression=[];classification=[];details=[]
    def evaluate(arm,model,seed,ids,pred):
        if len(ids)!=len(set(ids)) or set(ids)!=set(all_ids):raise ValueError('Incomplete or duplicate OOF targets')
        ix=np.array([positions[int(cid)] for cid in ids])
        reg,cls,_,detail=score(Y[ix],pred,[.5,1.])
        regression.append(dict(arm=arm,model=model,model_seed=seed,**reg))
        classification.extend([dict(arm=arm,model=model,model_seed=seed,**r) for r in cls])
        details.append(pd.DataFrame(dict(condition_id=ids,compound=meta.iloc[ix].compound.to_numpy(),
            response_mse=detail['rmse']**2,mae=detail['mae'],arm=arm,model=model,model_seed=seed)))
    for seed in seeds:
        for arm in a.arms:
            for model in a.families:
                ids=[];predictions=[]
                for fold in range(a.folds):
                    folder=a.out/f'fold_{fold}'/f'seed_{seed}'/arm/model
                    job=dict(split_seed=a.split_seed,fold=fold,model_seed=seed,model=model,arm=arm)
                    record=core.read_completion(folder,job)
                    if record is None:raise ValueError('Missing completion marker')
                    with np.load(folder/record['prediction_path'],allow_pickle=False) as saved:
                        ids.extend(saved['target_ids'].tolist());predictions.append(saved['prediction'])
                evaluate(arm,model,seed,np.asarray(ids),np.concatenate(predictions))
    ids=np.concatenate([seen_baselines[f][1] for f in range(a.folds)])
    mean=np.concatenate([seen_baselines[f][2] for f in range(a.folds)])
    evaluate('baseline','time_level_mean',-1,ids,mean)
    evaluate('baseline','zero_response',-1,ids,np.zeros_like(mean))
    reg=pd.DataFrame(regression);cls=pd.DataFrame(classification)
    reg.to_csv(a.out/'pooled_by_model_seed.csv',index=False)
    cls.to_csv(a.out/'classification_by_model_seed.csv',index=False)
    detail=pd.concat(details,ignore_index=True);detail.to_csv(a.out/'oof_condition_metrics.csv',index=False)
    summary=reg.groupby(['arm','model'])[['rmse','mae','pearson_target_mean','spearman_target_mean','r2_probe_variance_weighted']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(a.out/'summary.csv')
    metrics=['macro_f1','down_f1','up_f1','active_sign_accuracy','actual_active_fraction','predicted_active_fraction']
    csummary=cls.groupby(['arm','model','threshold'])[metrics].agg(['mean','std'])
    csummary.columns=['_'.join(c) for c in csummary.columns];csummary.to_csv(a.out/'classification_summary.csv')
    wide=reg[reg.model_seed.ge(0)].pivot(index='model_seed',columns=['arm','model'],values='rmse')
    paired=[]
    for seed,row in wide.iterrows():
        for model in a.families:
            if 'basic' in a.arms and 'morgan' in a.arms:
                paired.append(dict(model_seed=seed,comparison='morgan_minus_basic_'+model,
                    rmse_difference=row[('morgan',model)]-row[('basic',model)]))
        for arm in a.arms:
            for left,right in [('ddpm','mlp'),('ddpm','cgan')]:
                if left in a.families and right in a.families:
                    paired.append(dict(model_seed=seed,comparison=arm+'_'+left+'_minus_'+right,
                        rmse_difference=row[(arm,left)]-row[(arm,right)]))
    pd.DataFrame(paired).to_csv(a.out/'paired_rmse_differences.csv',index=False)
    core.save_json(a.out/'evaluation_status.json',dict(full_fold_model_seeds=seeds,
        expected_model_seeds=a.model_seeds,all_requested_units_completed=len(complete)==a.folds*len(a.model_seeds),
        fold_averages_not_used=True,prediction_ensembling=False,
        uncertainty='std_across_model_initializations_only_not_compound_sampling_CI'))
    print(summary.to_string(),flush=True)
    return len(complete)==a.folds*len(a.model_seeds)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','structure','smiles','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--model-seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--arms',nargs='+',choices=['basic','morgan'],default=['basic','morgan'])
    p.add_argument('--families',nargs='+',choices=['mlp','cgan','ddpm'],default=['mlp','cgan','ddpm'])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--budget-minutes',type=float,default=60.)
    p.add_argument('--unit-budget-minutes',type=float,default=15.)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if a.folds<2 or a.epochs<1 or a.threads<1 or min(a.budget_minutes,a.unit_budget_minutes)<=0:p.error('Invalid counts/budget')
    for values in [a.model_seeds,a.arms,a.families]:
        if len(values)!=len(set(values)):p.error('Duplicates')
    if any(s<0 or s>=2**32 for s in [a.split_seed]+a.model_seeds):p.error('Invalid seeds')
    jobs=a.folds*len(a.model_seeds)*len(a.arms)*len(a.families)
    print(json.dumps(dict(mode='execute' if a.execute else 'plan_only',jobs=jobs,fits_up_to=2*jobs,
        folds=a.folds,model_seeds=a.model_seeds,arms=a.arms,families=a.families,max_epochs=a.epochs,
        descriptor='excluded_pending_numeric_diagnosis',soft_global_budget=a.budget_minutes,
        note='Global budget checked between fold/seed units; current unit and evaluation can overrun'),indent=2),flush=True)
    if not a.execute:return jobs
    started=time.monotonic()
    core.protect_output(a.out,[a.prepared,a.structure,a.smiles])
    # Root contract prevents mixing schedules/settings when resuming.
    contract=dict(arguments={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='execute'},
        source_sha256={str(Path(__file__).resolve()):core.digest(Path(__file__))})
    core.contract_folder(a.out,contract)
    stopped=False
    for seed in a.model_seeds:
        for fold in range(a.folds):
            folder=a.out/f'fold_{fold}'/f'seed_{seed}'
            if not (folder/'completed.json').exists() and time.monotonic()-started>a.budget_minutes*60:
                stopped=True;break
            print(f'FOLD {fold+1}/{a.folds}, MODEL SEED {seed}',flush=True)
            # Invoking even completed units verifies their input/source/version contracts and hashes.
            pilot.main(arguments(a,fold,seed)+['--execute'])
            if not (folder/'completed.json').exists():stopped=True;break
        if stopped:break
    complete=summarize(a)
    core.save_json(a.out/'repeat_status.json',dict(all_complete=complete,elapsed_minutes=(time.monotonic()-started)/60,
                                                  interrupted_by_budget=stopped))
    print('REPEATED STUDY COMPLETE' if complete else 'PARTIAL RESULTS SAVED; resume with identical command',a.out,flush=True)
    return jobs


if __name__=='__main__':main()
