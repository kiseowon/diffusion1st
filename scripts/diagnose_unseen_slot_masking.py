"""Frozen DDPM counterfactual: mask only source slots absent during final training."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import repeated_latent_validation as core
import aivive_inspired_validation as inspired
import meeting_time_structure_pilot as pilot
import bounded_descriptors


def features(table):
    return pd.DataFrame(dict(dose=np.log1p(table.dose.to_numpy(float)),time=table.time_h.to_numpy(float),unit=table.unit.astype(str),level=table.level.astype(str)))


def reconstruct(unit, state, meta, prepared):
    raw=np.load(prepared/'vitro_expression.npy',mmap_mode='r')
    table=pd.read_csv(prepared/'vitro_conditions.csv')
    if not np.array_equal(table.condition_id,np.arange(len(raw))):raise ValueError('Source row alignment')
    audit=pd.read_csv(unit/'source_control_audit.csv')
    source=table.iloc[audit.condition_id.to_numpy(int)].reset_index(drop=True)
    delta=np.asarray(raw[audit.condition_id.to_numpy(int)]-raw[audit.control_condition_id.to_numpy(int)],dtype=np.float32)
    expression=state['source_scaler'].transform(delta)
    scores=state['source_score_scaler'].transform(state['source_pca'].transform(expression)[:,state['source_keep']])
    conditions=state['source_condition_encoder'].transform(features(source))
    token=np.c_[scores,conditions,np.ones(len(source))].astype(np.float32)
    slots=pd.DataFrame(state['source_slot_audit']).set_index('condition_id')['slot']
    assigned=source.condition_id.map(slots)
    if assigned.isna().any():raise ValueError('Missing source slot')
    assigned=assigned.to_numpy(int)
    split=pd.read_csv(unit/'compound_split.csv')
    train=set(split.loc[split.stage.isin(['train','validation']),'compound'])
    observed=set(assigned[source.compound.isin(train).to_numpy()])
    unseen=sorted(set(range(len(state['slots'])))-observed)
    sequences={}
    for compound,g in source.groupby('compound',sort=True):
        idx=g.index.to_numpy();matrix=np.zeros((len(state['slots']),token.shape[1]),dtype=np.float32)
        if len(set(assigned[idx]))!=len(idx):raise ValueError('Duplicate compound slots')
        matrix[assigned[idx]]=token[idx];sequences[compound]=matrix.ravel()
    target=state['target_condition_encoder'].transform(features(meta))
    base=np.c_[np.asarray([sequences[c] for c in meta.compound]),target].astype(np.float32)
    masked=base.copy()
    for slot in unseen:masked[:,slot*token.shape[1]:(slot+1)*token.shape[1]]=0
    changed=(base!=masked).any(1)
    return base,masked,changed,unseen,token.shape[1]


def append_structure(base, state, meta, unit, structure_path):
    arm=state['structure_representation'];width=state['structure_slots']
    if state.get('structure_layout')!='separate':raise ValueError('Expected separate chemical feature blocks')
    block=state['structure_block_width'];values=np.zeros((len(meta),width),dtype=np.float32)
    if arm in ['morgan','both']:
        raw=pd.read_csv(structure_path);raw.compound=raw.compound.astype(str).str.strip();raw=raw.set_index('compound')
        if raw.index.duplicated().any():raise ValueError('Duplicate Morgan names')
        values[:,:raw.shape[1]]=raw.reindex(meta.compound).fillna(0).to_numpy(np.float32)
        values[:,block-1]=meta.compound.isin(raw.index).to_numpy()
    if arm in ['descriptor','both']:
        raw=pd.read_csv(unit/'raw_descriptors.csv',index_col='compound')
        encoded,_=bounded_descriptors.encode(raw,meta.compound.tolist(),state['descriptor_preprocess'],block)
        values[:,block:]=encoded
    return torch.tensor(np.c_[base,values],dtype=torch.float32)


def diagnose(unit,out,arms,threads=4):
    import sklearn,scipy
    info=json.loads((unit/'run.json').read_text());settings=info['settings']
    prepared=Path(info['arguments']['prepared']);structure=Path(info['arguments']['structure'])
    core.protect_output(out,[unit,prepared,structure])
    if out.exists():raise ValueError('Use a fresh diagnostic folder')
    current=dict(numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,scipy=scipy.__version__)
    for name,version in current.items():
        if info['versions'][name]!=version:raise ValueError('Original environment required: '+name)
    for path,expected in info['input_sha256'].items():
        if core.digest(Path(path))!=expected:raise ValueError('Input checksum mismatch: '+path)
    for path in [Path(core.__file__),Path(inspired.__file__),Path(pilot.__file__),Path(bounded_descriptors.__file__)]:
        matches=[h for n,h in info['source_sha256'].items() if Path(n).name==path.name]
        if matches!=[core.digest(path)]:raise ValueError('Original prediction/preprocessing code differs: '+path.name)
    if settings.get('output_space')!='full' or settings.get('input_space')!='pca':raise ValueError('Requires saved input PCA/full output run')
    torch.set_num_threads(threads)
    target_table=pd.read_csv(prepared/'vivo_conditions.csv');actual=np.load(prepared/'vivo_expression.npy',mmap_mode='r')
    if not np.array_equal(target_table.condition_id,np.arange(len(actual))):raise ValueError('Target alignment')
    ta=pd.read_csv(unit/'target_control_audit.csv').set_index('condition_id').control_condition_id
    out.mkdir(parents=True,exist_ok=False)
    old=core.PROTOCOL.copy();rows=[];details=[];audit_rows=[];consumed=[unit/'run.json',unit/'raw_descriptors.csv',unit/'source_control_audit.csv',unit/'target_control_audit.csv',unit/'compound_split.csv']
    try:
        core.PROTOCOL.update(settings)
        for arm in arms:
            task=unit/arm/'ddpm';marker=task/'complete.json';record=json.loads(marker.read_text())
            for name,checksum in record['sha256'].items():
                if core.digest(task/name)!=checksum:raise ValueError('Saved artifact checksum mismatch: '+name)
            prediction_path=task/record['prediction_path'];model_path=prediction_path.parent/'model.pt'
            state_path=unit/arm/'final_preprocess.joblib';state=joblib.load(state_path)
            consumed.extend([marker,prediction_path,model_path,state_path])
            with np.load(prediction_path,allow_pickle=False) as saved:ids=saved['target_ids'].astype(int);saved_prediction=saved['prediction'].copy()
            if len(set(ids))!=len(ids):raise ValueError('Duplicate targets')
            meta=target_table.iloc[ids].reset_index(drop=True)
            base,masked,changed,unseen,token_width=reconstruct(unit,state,meta,prepared)
            X=append_structure(base,state,meta,unit,structure);MX=append_structure(masked,state,meta,unit,structure)
            seed=record['job']['model_seed']
            model,_,_,_=inspired.models('ddpm',X.shape[1],len(state['output_mean']),seed)
            unseen_columns=[slot*token_width+j for slot in unseen for j in range(token_width)]
            initial_unseen=model.context[0].weight[:,unseen_columns].detach().clone()
            model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True))
            weight_change=float((model.context[0].weight[:,unseen_columns].detach()-initial_unseen).abs().max()) if unseen_columns else 0.
            indices=np.arange(len(ids));noise_seed=seed+30000
            regenerated=pilot.restore_output(core.predict_family_scores('ddpm',model,{'X':X},indices,noise_seed),state)
            max_error=float(np.max(np.abs(regenerated-saved_prediction)))
            if not np.allclose(regenerated,saved_prediction,rtol=5e-6,atol=1e-6):
                core.save_json(out/'reproduction_failed.json',dict(arm=arm,max_abs_difference=max_error,masking_not_run=True))
                raise ValueError('Original prediction failed reproduction; stopped before masking: '+arm)
            counterfactual=pilot.restore_output(core.predict_family_scores('ddpm',model,{'X':MX},indices,noise_seed),state)
            truth=np.asarray(actual[ids]-actual[ta.loc[ids].to_numpy(int)],dtype=np.float64)
            original_mse=((regenerated-truth)**2).mean(1);masked_mse=((counterfactual-truth)**2).mean(1)
            if not np.isfinite(counterfactual).all():raise ValueError('Nonfinite masked predictions')
            rows.append(dict(arm=arm,model_seed=seed,epoch=record['selection']['epoch'],noise_seed=noise_seed,reproduction_max_abs_difference=max_error,original_rmse=float(np.sqrt(original_mse.mean())),masked_rmse=float(np.sqrt(masked_mse.mean())),changed_targets=int(changed.sum()),unseen_slots=len(unseen),unseen_weight_max_change_from_seed_initialization=weight_change,original_prediction_abs_max=float(np.abs(regenerated).max()),masked_prediction_abs_max=float(np.abs(counterfactual).max())))
            for c,g in meta.groupby('compound'):
                ix=g.index.to_numpy();details.append(dict(arm=arm,compound=c,changed_input=bool(changed[ix].any()),n_targets=len(ix),original_rmse=float(np.sqrt(original_mse[ix].mean())),masked_rmse=float(np.sqrt(masked_mse[ix].mean()))))
            for slot in unseen:audit_rows.append(dict(arm=arm,slot=slot,definition=repr(state['slots'][slot]),token_width=token_width))
            np.savez_compressed(out/f'{arm}_diagnostic_predictions.npz',target_ids=ids,original=regenerated,masked=counterfactual,changed_input=changed)
            pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);pd.DataFrame(details).to_csv(out/'by_compound.csv',index=False)
            print(rows[-1],flush=True)
    finally:core.PROTOCOL.clear();core.PROTOCOL.update(old)
    pd.DataFrame(audit_rows).to_csv(out/'masked_slots.csv',index=False)
    core.save_json(out/'provenance.json',dict(unit=str(unit.resolve()),arms=arms,threads=threads,training=False,refitting_preprocessing=False,prediction_regeneration=True,same_noise_seed=True,scope='Post-hoc frozen-model input ablation, not a corrected benchmark or independent test; masked information is removed.',consumed_sha256={str(p.resolve()):core.digest(p) for p in consumed},input_sha256=info['input_sha256'],diagnostic_sha256=core.digest(Path(__file__)),prediction_code_hash_verified=True))
    print('\nSUMMARY\n'+pd.DataFrame(rows).to_string(index=False),flush=True)
    print('\nCOMPOUND DETAILS\n'+pd.DataFrame(details).sort_values(['arm','original_rmse'],ascending=[True,False]).to_string(index=False),flush=True)
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--unit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--arms',nargs='+',choices=['basic','morgan','descriptor','both'],default=['basic','morgan','descriptor','both']);a=p.parse_args()
    if a.threads<1 or len(a.arms)!=len(set(a.arms)):p.error('Invalid threads/duplicate arms')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=a.threads):diagnose(a.unit,a.out,a.arms,a.threads)

if __name__=='__main__':main()
