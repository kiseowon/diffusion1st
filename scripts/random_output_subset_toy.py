"""Fixed random output-probe subset versus saved full-output toy models on identical probes."""
import argparse
import json
import platform
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
import torch
import joblib
import repeated_latent_validation as core
import fixed_condition_toy as toy
import meeting_time_structure_pilot as pilot
import aivive_inspired_validation as inspired
import bounded_descriptors
from diagnose_toy_fit import reconstruct
from evaluate_repeated_predictions import score


def selected_indices(total,count,seed):
    if not 1<=count<=total:raise ValueError('Subset size must be within available probe count')
    return np.sort(np.random.default_rng(seed).choice(total,size=count,replace=False))


def subset_pack(X,Y,state,fit,indices):
    mean=np.asarray(state['output_mean'])[indices].copy()
    # Retain full-model shared RMS: do not introduce a new target scaling policy.
    scale=state['output_scale'];Z=torch.zeros((len(Y),len(indices)),dtype=torch.float32)
    Z[fit]=torch.tensor((Y[fit][:,indices]-mean)/scale,dtype=torch.float32)
    transformed_state=dict(state,output_mean=mean,output_scale=scale,output_dimension=len(indices),selected_probe_indices=indices.copy(),target_scaling='mean sliced to selected probes, shared RMS retained from full-output reference')
    if 'target_roundtrip_max_abs_error' in transformed_state:transformed_state['reference_full_transform_roundtrip_max_abs_error']=transformed_state.pop('target_roundtrip_max_abs_error')
    return dict(X=X,Z=Z,state=transformed_state)


def verify_reference(run):
    import sklearn,scipy
    if not json.loads((run/'complete.json').read_text()).get('all_complete'):raise ValueError('Completed fixed-condition toy reference required')
    info=json.loads((run/'run.json').read_text())
    if info['settings'].get('protocol')!='fixed_24h_High_toy_v1':raise ValueError('Unsupported reference protocol')
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,joblib=joblib.__version__)
    if any(info['versions'][k]!=v for k,v in versions.items()):raise ValueError('Use the reference environment')
    for path,checksum in info['input_sha256'].items():
        if core.digest(Path(path))!=checksum:raise ValueError('Reference input changed: '+path)
    for path in [Path(m.__file__) for m in [toy,core,pilot,inspired,bounded_descriptors]]+[Path(__file__).with_name('evaluate_repeated_predictions.py')]:
        matches=[v for n,v in info['source_sha256'].items() if Path(n).name==path.name]
        if matches!=[core.digest(path)]:raise ValueError('Shared model/preprocessing code differs: '+path.name)
    return info


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n-probes',type=int,default=256);p.add_argument('--probe-seed',type=int,default=20261109);p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if a.n_probes<1 or not 0<=a.probe_seed<2**32:p.error('Invalid subset size or probe seed')
    print(json.dumps(dict(mode='execute' if a.execute else 'plan_only',n_probes=a.n_probes,probe_seed=a.probe_seed,default_reference_jobs=27,epoch_selection='fixed separately for each arm/model/model-seed from completed full-output reference',scope='same reference holdout and input features; full and subset models evaluated on identical selected probes; no full-model retraining'),indent=2),flush=True)
    if not a.execute:return
    run=a.reference_run.resolve();info=verify_reference(run);args=info['arguments'];settings=info['settings']
    prepared=Path(args['prepared']);structure=Path(args['structure'])
    core.protect_output(a.out,[run,prepared,structure,Path(args['smiles'])])
    table=pd.read_csv(run/'cohort.csv');split=pd.read_csv(run/'compound_split.csv')
    if not table.compound.is_unique or not table[['compound','source_condition_id','target_condition_id']].equals(split[['compound','source_condition_id','target_condition_id']]):raise ValueError('Cohort/split mismatch')
    itr=np.flatnonzero(split.stage.eq('train'));val=np.flatnonzero(split.stage.eq('validation'));test=np.flatnonzero(split.stage.eq('test'));train=np.sort(np.r_[itr,val])
    if min(map(len,[itr,val,test]))<1 or len(train)+len(test)!=len(table):raise ValueError('Invalid partitions')
    probes=pd.read_csv(prepared/'probes.csv')['probe']
    if not probes.is_unique:raise ValueError('Duplicate probe identities')
    indices=selected_indices(len(probes),a.n_probes,a.probe_seed)
    reference_files=[run/n for n in ['run.json','complete.json','cohort.csv','compound_split.csv','source_control_audit.csv','target_control_audit.csv','raw_descriptors.csv','baseline_predictions.npz']]
    for arm in ['basic','morgan','descriptor']:
        reference_files.extend(run/arm/n for n in ['inner_preprocess.joblib','final_preprocess.joblib'])
        for seed in args['model_seeds']:
            for kind in ['mlp','cgan','ddpm']:reference_files.append(run/arm/f'seed_{seed}'/kind/'complete.json')
    source_files=list(Path(__file__).parent.glob('*.py'))+[core.SNAPSHOT]
    contract=dict(arguments={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='execute'},reference_sha256={str(f.resolve()):core.digest(f) for f in reference_files},input_sha256=info['input_sha256'],source_sha256={str(f.resolve()):core.digest(f) for f in source_files},versions=info['versions'],selected_probe_indices=indices.tolist(),epoch_policy='fixed full-output reference epoch; subset validation is recorded but not used to select a new epoch',limitations='Output head/noisy-target dimensions and parameter count differ. Training loss now covers selected probes. Same seed labels do not imply identical random draws across different network dimensions. Single random subset and exploratory holdout, not independent validation or a probe-count optimum.')
    core.contract_folder(a.out,contract)
    pd.DataFrame(dict(original_probe_index=indices,probe=probes.iloc[indices].to_numpy())).to_csv(a.out/'selected_probes.csv',index=False)
    table.to_csv(a.out/'cohort.csv',index=False);split.to_csv(a.out/'compound_split.csv',index=False)
    torch.set_num_threads(args['threads']);regression=[];classification=[];paired=[]
    old=core.PROTOCOL.copy();hooks={k:getattr(core,k) for k in ['make_family_model','train_family_epoch','restore']}
    def evaluate(mode,arm,kind,seed,truth,pred):
        reg,cls,_,_=score(truth,pred,[.5,1.])
        regression.append(dict(output_training=mode,arm=arm,model=kind,model_seed=seed,**reg))
        classification.extend([dict(output_training=mode,arm=arm,model=kind,model_seed=seed,**r) for r in cls])
        return reg
    from threadpoolctl import threadpool_limits
    total=9*len(args['model_seeds']);done=0;common_truth=None
    try:
        core.PROTOCOL.update(settings);core.make_family_model=toy.models;core.train_family_epoch=inspired.train;core.restore=pilot.restore_output
        with threadpool_limits(limits=args['threads']):
            for arm in ['basic','morgan','descriptor']:
                inner_state=joblib.load(run/arm/'inner_preprocess.joblib');final_state=joblib.load(run/arm/'final_preprocess.joblib')
                iX,Y=reconstruct(run,inner_state,table,prepared,structure);oX,Y2=reconstruct(run,final_state,table,prepared,structure)
                if Y.shape[1]!=len(probes) or not np.array_equal(Y,Y2):raise ValueError('Reference target/probe alignment')
                if common_truth is None:common_truth=Y
                elif not np.array_equal(common_truth,Y):raise ValueError('Targets differ across structure arms')
                inner=subset_pack(iX,Y,inner_state,itr,indices);outer=subset_pack(oX,Y,final_state,train,indices);small_Y=Y[:,indices]
                folder=a.out/arm;folder.mkdir(exist_ok=True)
                joblib.dump(inner['state'],folder/'inner_preprocess.joblib');joblib.dump(outer['state'],folder/'final_preprocess.joblib')
                for seed in args['model_seeds']:
                    for kind in ['mlp','cgan','ddpm']:
                        original_task=run/arm/f'seed_{seed}'/kind
                        original=core.read_completion(original_task,dict(model=kind,model_seed=seed,arm=arm,split_seed=args['split_seed'],fold=args['fold']))
                        if original is None:raise ValueError('Missing reference condition')
                        epoch=int(original['selection']['epoch'])
                        if epoch<1:raise ValueError('Expected trained direct-response reference')
                        with np.load(original_task/original['prediction_path']) as saved:
                            if not np.array_equal(saved['target_ids'],table.iloc[test].target_condition_id):raise ValueError('Reference test target mismatch')
                            full_pred=saved['prediction'][:,indices].copy()
                        full_metrics=evaluate('full_model_on_same_selected_probes',arm,kind,seed,small_Y[test],full_pred)
                        task=folder/f'seed_{seed}'/kind;task.mkdir(parents=True,exist_ok=True)
                        job=dict(model=kind,model_seed=seed,arm=arm,split_seed=args['split_seed'],fold=args['fold'],n_probes=a.n_probes,probe_seed=a.probe_seed)
                        record=core.read_completion(task,job)
                        if record is None:
                            attempt=task/('attempt_'+uuid.uuid4().hex);attempt.mkdir();started=time.monotonic()
                            core.PROTOCOL.update(epochs=epoch,validation_epochs=[epoch])
                            model,_,disc,_=toy.models(kind,inner['X'].shape[1],a.n_probes,seed)
                            reference_dimensions=json.loads((original_task/original['prediction_path']).parent.joinpath('model_dimensions.json').read_text())
                            core.save_json(attempt/'model_dimensions.json',dict(input_dimension=inner['X'].shape[1],output_dimension=a.n_probes,model_parameters=sum(p.numel() for p in model.parameters()),discriminator_parameters=0 if disc is None else sum(p.numel() for p in disc.parameters()),full_output_reference=reference_dimensions))
                            del model,disc
                            print('START',job,'fixed reference epoch',epoch,flush=True)
                            offset=np.zeros_like(small_Y)
                            pred,selection=core.fit_job(job,inner,outer,small_Y,itr,val,test,train,offset,offset,attempt)
                            selection.update(reference_epoch=epoch,epoch_policy='fixed_from_original_full_output_validation',reference_validation_rmse_all_probes=original['selection']['validation_response_rmse'],elapsed_seconds=time.monotonic()-started)
                            core.save_json(attempt/'selection.json',selection)
                            np.savez_compressed(attempt/'predictions.npz',target_ids=table.iloc[test].target_condition_id.to_numpy(),probe_indices=indices,prediction=pred)
                            record=dict(job=job,prediction_path=str((attempt/'predictions.npz').relative_to(task)),selection=selection,reference_prediction_sha256=core.digest(original_task/original['prediction_path']),sha256={str(f.relative_to(task)):core.digest(f) for f in attempt.iterdir() if f.is_file()})
                            core.save_json(task/'complete.json',record)
                        if record['reference_prediction_sha256']!=core.digest(original_task/original['prediction_path']):raise ValueError('Reference predictions changed on resume')
                        with np.load(task/record['prediction_path']) as saved:
                            if not np.array_equal(saved['target_ids'],table.iloc[test].target_condition_id) or not np.array_equal(saved['probe_indices'],indices):raise ValueError('Subset result identities mismatch')
                            pred=saved['prediction']
                        subset_metrics=evaluate('subset_model',arm,kind,seed,small_Y[test],pred)
                        paired.append(dict(arm=arm,model=kind,model_seed=seed,reference_epoch=epoch,full_rmse_same_probes=full_metrics['rmse'],subset_rmse=subset_metrics['rmse'],subset_minus_full_rmse=subset_metrics['rmse']-full_metrics['rmse'],full_mae_same_probes=full_metrics['mae'],subset_mae=subset_metrics['mae'],full_pearson_same_probes=full_metrics['pearson_target_mean'],subset_pearson=subset_metrics['pearson_target_mean']))
                        done+=1;core.save_json(a.out/'progress.json',dict(completed=done,total=total));print('COMPLETE',done,'/',total,'subset-full RMSE',paired[-1]['subset_minus_full_rmse'],flush=True)
                        pd.DataFrame(regression).to_csv(a.out/'by_seed.csv',index=False);pd.DataFrame(paired).to_csv(a.out/'paired_by_seed.csv',index=False)
            with np.load(run/'baseline_predictions.npz') as saved:
                if not np.array_equal(saved['target_ids'],table.iloc[test].target_condition_id):raise ValueError('Baseline targets mismatch')
                baseline=saved['training_mean'][:,indices]
            if not np.allclose(baseline,common_truth[train][:,indices].mean(0)[None,:],rtol=0,atol=1e-12):raise ValueError('Training mean baseline mismatch')
            evaluate('baseline','baseline','training_compound_mean',-1,common_truth[test][:,indices],baseline)
            evaluate('baseline','baseline','zero_response',-1,common_truth[test][:,indices],np.zeros_like(baseline))
    finally:
        core.PROTOCOL.clear();core.PROTOCOL.update(old)
        for name,hook in hooks.items():setattr(core,name,hook)
    reg=pd.DataFrame(regression);cls=pd.DataFrame(classification)
    reg.to_csv(a.out/'by_seed.csv',index=False);cls.to_csv(a.out/'classification_by_seed.csv',index=False)
    summary=reg.groupby(['output_training','arm','model'])[['rmse','mae','pearson_target_mean','spearman_target_mean','r2_probe_variance_weighted']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(a.out/'summary.csv')
    cs=cls.groupby(['output_training','arm','model','threshold'])[['macro_f1','down_f1','up_f1','active_sign_accuracy','actual_active_fraction','predicted_active_fraction']].agg(['mean','std']);cs.columns=['_'.join(c) for c in cs.columns];cs.to_csv(a.out/'classification_summary.csv')
    comparison=pd.DataFrame(paired).groupby(['arm','model']).agg(full_rmse_same_probes_mean=('full_rmse_same_probes','mean'),subset_rmse_mean=('subset_rmse','mean'),subset_minus_full_mean=('subset_minus_full_rmse','mean'),subset_minus_full_std=('subset_minus_full_rmse','std'))
    comparison.to_csv(a.out/'comparison_summary.csv')
    core.save_json(a.out/'complete.json',dict(all_complete=True,jobs=total,n_selected_probes=a.n_probes,probe_seed=a.probe_seed,one_probe_subset_only=True,whole_reference_probes=len(probes),reference_models_retrained=False,all_folds_evaluated=False,epoch_policy='fixed_reference_epochs',scope='exploratory paired output-size pipeline comparison; model capacity also changes'))
    print('\nSAME-PROBE COMPARISON\n'+comparison.to_string(),flush=True)
    print('COMPLETE',a.out,flush=True)
    return total

if __name__=='__main__':main()
