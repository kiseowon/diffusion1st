"""Frozen final-model train/test and training-denoising diagnostics for the 24h High toy."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import repeated_latent_validation as core
import fixed_condition_toy as toy
import meeting_time_structure_pilot as pilot
import bounded_descriptors
import aivive_inspired_validation as inspired
from evaluate_repeated_predictions import score
from diagnose_unseen_slot_masking import reproduction_report


def reconstruct(run,state,table,prepared,structure):
    raw_source=np.load(prepared/'vitro_expression.npy',mmap_mode='r')
    raw_target=np.load(prepared/'vivo_expression.npy',mmap_mode='r')
    sa=pd.read_csv(run/'source_control_audit.csv').set_index('condition_id').control_condition_id
    ta=pd.read_csv(run/'target_control_audit.csv').set_index('condition_id').control_condition_id
    sids=table.source_condition_id.to_numpy(int);tids=table.target_condition_id.to_numpy(int)
    delta=np.asarray(raw_source[sids]-raw_source[sa.loc[sids].to_numpy(int)],dtype=np.float32)
    truth=np.asarray(raw_target[tids]-raw_target[ta.loc[tids].to_numpy(int)],dtype=np.float64)
    source=state['source_scaler'].transform(delta)
    e=state['source_score_scaler'].transform(state['source_pca'].transform(source)).astype(np.float32)
    values=np.zeros((len(table),state['structure_width']),np.float32)
    arm=state['structure_representation']
    if arm=='morgan':
        st=pd.read_csv(structure);st.compound=st.compound.astype(str).str.strip();st=st.set_index('compound')
        m=st.reindex(table.compound).to_numpy(np.float32)
        if not np.isfinite(m).all():raise ValueError('Missing Morgan in saved common cohort')
        values[:,:m.shape[1]]=m
    elif arm=='descriptor':
        raw=pd.read_csv(run/'raw_descriptors.csv',index_col='compound',float_precision='round_trip')
        values=bounded_descriptors.encode(raw,table.compound.tolist(),state['descriptor_preprocess'],state['structure_width']+1)[0][:,:-1]
    X=torch.tensor(np.c_[e,values],dtype=torch.float32)
    if not torch.isfinite(X).all() or not np.isfinite(truth).all():raise ValueError('Nonfinite diagnostic data')
    return X,truth


def denoising(model,X,truth,state,noise_seed):
    """Teacher-forced training-target diagnostic, never a prediction metric."""
    _,_,abar,_=core.schedule();rows=[]
    steps=sorted(set([0,core.PROTOCOL['steps']//4,core.PROTOCOL['steps']//2,3*core.PROTOCOL['steps']//4,core.PROTOCOL['steps']-1]))
    clean=torch.tensor((truth-state['output_mean'])/state['output_scale'],dtype=torch.float32)
    model.eval()
    for step in steps:
        generator=torch.Generator().manual_seed(noise_seed+step)
        noise=torch.randn(clean.shape,generator=generator)
        noisy=abar[step].sqrt()*clean+(1-abar[step]).sqrt()*noise
        predictions=[]
        with torch.no_grad():
            for start in range(0,len(X),core.PROTOCOL['batch_size']):
                stop=min(len(X),start+core.PROTOCOL['batch_size'])
                predictions.append(model(X[start:stop],noisy[start:stop],torch.full((stop-start,),step,dtype=torch.long)))
        z=torch.cat(predictions).numpy()
        if not np.isfinite(z).all():raise ValueError('Nonfinite teacher-forced denoising outputs')
        restored=pilot.restore_output(z,state)
        rows.append(dict(step=step,alpha_bar=float(abar[step]),scaled_x0_mse=float(np.mean((z-clean.numpy())**2)),scaled_training_mean_baseline_mse=float(clean.square().mean()),probe_reconstruction_rmse=core.rmse(truth,restored),uses_true_training_target_in_noisy_input=True,evaluation='teacher_forced_final_training_denoising_not_free_generation'))
    return rows


def diagnose(run,out):
    import sklearn,scipy
    if not json.loads((run/'complete.json').read_text()).get('all_complete'):raise ValueError('Completed toy run required')
    info=json.loads((run/'run.json').read_text());arguments=info['arguments'];settings=info['settings']
    if settings.get('protocol')!='fixed_24h_High_toy_v1':raise ValueError('Not the supported toy protocol')
    prepared=Path(arguments['prepared']);structure=Path(arguments['structure'])
    core.protect_output(out,[run,prepared,structure])
    if out.exists():raise ValueError('Use a fresh diagnostic output')
    current=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,scipy=scipy.__version__)
    if any(info['versions'][k]!=v for k,v in current.items()):raise ValueError('Use original training environment')
    for path,checksum in info['input_sha256'].items():
        if core.digest(Path(path))!=checksum:raise ValueError('Original input changed: '+path)
    for path in [Path(m.__file__) for m in [toy,core,pilot,inspired,bounded_descriptors]]+[Path(__file__).with_name('evaluate_repeated_predictions.py')]:
        matches=[v for n,v in info['source_sha256'].items() if Path(n).name==path.name]
        if matches!=[core.digest(path)]:raise ValueError('Training/prediction source differs: '+path.name)
    table=pd.read_csv(run/'cohort.csv');split=pd.read_csv(run/'compound_split.csv')
    if not table.compound.is_unique or not table[['compound','source_condition_id','target_condition_id']].equals(split[['compound','source_condition_id','target_condition_id']]):raise ValueError('Cohort/split mismatch')
    train=np.flatnonzero(split.stage.isin(['train','validation']));test=np.flatnonzero(split.stage.eq('test'))
    if not len(train) or not len(test) or set(train)&set(test) or len(train)+len(test)!=len(table):raise ValueError('Invalid final training/test partition')
    torch.set_num_threads(arguments['threads']);out.mkdir(parents=True,exist_ok=False)
    consumed=[run/n for n in ['run.json','complete.json','cohort.csv','compound_split.csv','source_control_audit.csv','target_control_audit.csv','raw_descriptors.csv']]
    metrics=[];classes=[];details=[];denoise_rows=[];old=core.PROTOCOL.copy()
    from threadpoolctl import threadpool_limits
    try:
        core.PROTOCOL.update(settings)
        with threadpool_limits(limits=arguments['threads']):
            for arm in ['basic','morgan','descriptor']:
                state_path=run/arm/'final_preprocess.joblib';state=joblib.load(state_path);consumed.append(state_path)
                X,Y=reconstruct(run,state,table,prepared,structure)
                baseline_mean=Y[train].mean(0)
                if not np.allclose(baseline_mean,state['output_mean'],rtol=0,atol=1e-12):raise ValueError('Saved output mean does not match final training cohort')
                for seed in arguments['model_seeds']:
                    for kind in ['mlp','cgan','ddpm']:
                        task=run/arm/f'seed_{seed}'/kind
                        record=core.read_completion(task,dict(model=kind,model_seed=seed,arm=arm,split_seed=arguments['split_seed'],fold=arguments['fold']))
                        if record is None:raise ValueError('Missing completed model')
                        p=task/record['prediction_path'];checkpoint=p.parent/'model.pt';consumed.extend([task/'complete.json',p,checkpoint])
                        model,_,_,_=toy.models(kind,X.shape[1],Y.shape[1],seed)
                        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
                        # Reproduce original test sampling before any train evaluation.
                        noise_seed=seed+30000
                        test_prediction=pilot.restore_output(core.predict_family_scores(kind,model,{'X':X},test,noise_seed),state)
                        with np.load(p) as saved:
                            if not np.array_equal(saved['target_ids'],table.iloc[test].target_condition_id):raise ValueError('Saved test IDs differ')
                            check=reproduction_report(saved['prediction'],test_prediction,Y[test],True)
                        tag=f'{arm}_{kind}_seed_{seed}';core.save_json(out/f'{tag}_reproduction.json',check)
                        if not check['accepted']:raise ValueError('Original test prediction not reproduced within recorded limits: '+tag)
                        train_prediction=pilot.restore_output(core.predict_family_scores(kind,model,{'X':X},train,noise_seed),state)
                        np.savez_compressed(out/f'{tag}_predictions.npz',train_target_ids=table.iloc[train].target_condition_id.to_numpy(),train_prediction=train_prediction,test_target_ids=table.iloc[test].target_condition_id.to_numpy(),test_prediction=test_prediction)
                        for phase,indices,pred in [('final_training',train,train_prediction),('test',test,test_prediction)]:
                            truth=Y[indices];reg,cls,_,_=score(truth,pred,[.5,1.])
                            mean=np.repeat(baseline_mean[None,:],len(indices),axis=0);base_rmse=core.rmse(truth,mean);base_mae=float(np.abs(truth-mean).mean())
                            variation=float(np.sqrt(np.mean((pred-pred.mean(0))**2)));truth_variation=float(np.sqrt(np.mean((truth-truth.mean(0))**2)))
                            metrics.append(dict(arm=arm,model=kind,model_seed=seed,epoch=record['selection']['epoch'],phase=phase,noise_seed=noise_seed,selected_inner_validation_rmse=record['selection']['validation_response_rmse'],**reg,training_mean_baseline_rmse=base_rmse,training_mean_baseline_mae=base_mae,zero_baseline_rmse=core.rmse(truth,np.zeros_like(truth)),skill_vs_training_mean=1-reg['rmse']**2/base_rmse**2 if base_rmse else np.nan,prediction_between_compound_rms=variation,truth_between_compound_rms=truth_variation,prediction_to_truth_variation_ratio=variation/truth_variation if truth_variation else np.nan))
                            classes.extend([dict(arm=arm,model=kind,model_seed=seed,phase=phase,**r) for r in cls])
                            per_target=np.mean((pred-truth)**2,axis=1)
                            details.extend([dict(arm=arm,model=kind,model_seed=seed,phase=phase,compound=c,rmse=float(np.sqrt(v))) for c,v in zip(table.iloc[indices].compound,per_target)])
                        if kind=='ddpm':denoise_rows.extend([dict(arm=arm,model_seed=seed,epoch=record['selection']['epoch'],**r) for r in denoising(model,X[train],Y[train],state,seed+60000)])
                        pd.DataFrame(metrics).to_csv(out/'fit_metrics.csv',index=False);pd.DataFrame(details).to_csv(out/'by_compound.csv',index=False)
                        print('EVALUATED',tag,'train RMSE',metrics[-2]['rmse'],'test RMSE',metrics[-1]['rmse'],flush=True)
    finally:core.PROTOCOL.clear();core.PROTOCOL.update(old)
    table_metrics=pd.DataFrame(metrics)
    summary=table_metrics.groupby(['arm','model','phase'])[['rmse','mae','pearson_target_mean','training_mean_baseline_rmse','training_mean_baseline_mae','skill_vs_training_mean','prediction_to_truth_variation_ratio']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(out/'summary.csv')
    pd.DataFrame(classes).to_csv(out/'classification.csv',index=False)
    pd.DataFrame(denoise_rows).to_csv(out/'ddpm_training_denoising.csv',index=False)
    core.save_json(out/'provenance.json',dict(run=str(run.resolve()),training=False,refitting_preprocessing=False,prediction_regeneration=True,train_partition='final train plus prior inner-validation compounds; in-sample for final refit',test_partition='original held-out compounds',teacher_forcing='true final-training responses only, separate diagnostic, never test prediction',uncertainty='model seeds also change generation noise; variation of generated predictions includes sampling variation',limitations='An in-sample/out-of-sample gap alone does not prove overfitting; target distributions may differ. Inner validation refers to a different model/preprocessing stage. No seed selection, retraining or corrected benchmark.',consumed_sha256={str(p.resolve()):core.digest(p) for p in consumed},diagnostic_sha256=core.digest(Path(__file__))))
    print('\nFINAL MODEL TRAIN/TEST\n'+summary.to_string(),flush=True)
    if denoise_rows:print('\nDDPM TEACHER-FORCED TRAINING DENOISING (not free-generation prediction)\n'+pd.DataFrame(denoise_rows).groupby(['arm','step'])[['scaled_x0_mse','scaled_training_mean_baseline_mse']].mean().to_string(),flush=True)
    return table_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();diagnose(a.run,a.out)

if __name__=='__main__':main()
