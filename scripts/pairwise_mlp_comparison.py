"""Pairwise IVIVE MLP, compared against a completed four-input MLP reference."""
import argparse
import json
import platform
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import joblib
import repeated_latent_validation as core
import meeting_time_structure_pilot as pilot
import aivive_inspired_validation as inspired
from evaluate_repeated_predictions import score


def condition_features(table):
    return pd.DataFrame(dict(dose=np.log1p(table.dose.to_numpy(float)),time=table.time_h.to_numpy(float),unit=table.unit.astype(str),level=table.level.astype(str)))


class PairFeatures:
    def __init__(self,source,source_index,target,target_index,structure,compound_index):
        self.source=source;self.source_index=source_index;self.target=target;self.target_index=target_index;self.structure=structure;self.compound_index=compound_index
        self.shape=(len(source_index),source.shape[1]+target.shape[1]+structure.shape[1])
    def __getitem__(self,rows):
        return torch.cat([self.source[self.source_index[rows]],self.target[self.target_index[rows]],self.structure[self.compound_index[rows]]],dim=1)


def prepare(data,source,ids,meta,Y,state,fit,arm):
    if state.get('source_input','present')!='present' or state.get('output_space')!='full' or state.get('input_space')!='pca':raise ValueError('Reference must use source-present/PCA/full-output')
    if state.get('structure_layout')!='separate':raise ValueError('Expected separate chemical slots')
    source_ids=source.condition_id.to_numpy(int)
    expression=state['source_scaler'].transform(data.source[source_ids])
    values=state['source_score_scaler'].transform(state['source_pca'].transform(expression)[:,state['source_keep']])
    scond=state['source_condition_encoder'].transform(condition_features(source))
    tcond=state['target_condition_encoder'].transform(condition_features(meta))
    slookup={int(cid):i for i,cid in enumerate(source_ids)};tlookup={int(cid):i for i,cid in enumerate(ids)}
    pair_rows=np.flatnonzero(np.isin(data.target_ids,ids))
    si=np.array([slookup[int(c)] for c in data.source_ids[pair_rows]])
    ti=np.array([tlookup[int(c)] for c in data.target_ids[pair_rows]])
    ci=data.compound_indices[pair_rows]
    width=state['structure_slots'];block=state['structure_block_width']
    structure=np.zeros((len(data.names),width),np.float32)
    if arm=='morgan':
        structure[:,:data.structure.shape[1]-1]=data.structure[:,:-1]
        structure[:,block-1]=data.structure[:,-1]
    pack=dict(X=PairFeatures(torch.tensor(np.c_[values,scond],dtype=torch.float32),torch.tensor(si),torch.tensor(tcond,dtype=torch.float32),torch.tensor(ti),torch.tensor(structure),torch.tensor(ci)),state=state,target_index=ti,pair_source_ids=data.source_ids[pair_rows],pair_target_ids=data.target_ids[pair_rows])
    z=torch.zeros(Y.shape,dtype=torch.float32)
    z[fit]=torch.tensor((Y[fit]-state['output_mean'])/state['output_scale'],dtype=torch.float32)
    pack['Z']=z
    return pack


def training_weights(target_index,fit):
    rows=np.flatnonzero(np.isin(target_index,fit))
    counts=np.bincount(target_index[rows],minlength=int(target_index.max())+1)
    if not np.all(counts[fit]>0):raise ValueError('Missing training target pairs')
    weights=np.zeros(len(target_index),np.float32)
    weights[rows]=(1/counts[target_index[rows]])*(len(rows)/len(fit))
    return rows,torch.tensor(weights)


def train_epoch(model,opt,pack,fit):
    import torch.nn.functional as F
    model.train();pair_rows,weights=training_weights(pack['target_index'],fit)
    # Match reference target-row epoch's number of optimizer updates.
    # Uniform pair sampling + inverse pair-count weights is an unbiased
    # estimator of the target-equal, source-averaged per-pair MAE objective.
    sampled=pair_rows[torch.randint(len(pair_rows),(len(fit),)).numpy()]
    for batch in torch.tensor(sampled).split(core.PROTOCOL['batch_size']):
        rows=batch.numpy();truth=pack['Z'][pack['target_index'][rows]]
        pred=model(pack['X'][rows],torch.zeros_like(truth),torch.zeros(len(rows),dtype=torch.long))
        loss=(F.l1_loss(pred,truth,reduction='none').mean(1)*weights[rows]).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite pair loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),core.PROTOCOL['gradient_clip']);opt.step()


def predict(model,pack,targets):
    lookup={int(t):i for i,t in enumerate(targets)}
    rows=np.flatnonzero(np.isin(pack['target_index'],targets))
    sums=np.zeros((len(targets),pack['Z'].shape[1]),np.float64);counts=np.zeros(len(targets),int)
    for start in range(0,len(rows),core.PROTOCOL['batch_size']):
        batch=rows[start:start+core.PROTOCOL['batch_size']]
        pred=pilot.restore_output(core.predict_family_scores('mlp',model,pack,batch,0),pack['state'])
        positions=np.array([lookup[int(t)] for t in pack['target_index'][batch]])
        np.add.at(sums,positions,pred);np.add.at(counts,positions,1)
    if not np.all(counts>0):raise ValueError('Missing query target pairs')
    result=sums/counts[:,None]
    if not np.isfinite(result).all():raise ValueError('Nonfinite target predictions')
    return result,counts


def fit(inner,outer,Y,itr,val,train,test,seed,folder):
    model,opt,_,_=inspired.models('mlp',inner['X'].shape[1],Y.shape[1],seed)
    history=[];best=None
    for epoch in range(1,core.PROTOCOL['epochs']+1):
        train_epoch(model,opt,inner,itr)
        if epoch in core.PROTOCOL['validation_epochs']:
            pred,_=predict(model,inner,val);value=core.rmse(Y[val],pred)
            row=dict(epoch=epoch,validation_response_rmse=value);history.append(row)
            if best is None or value<best['validation_response_rmse']:best=row.copy()
            pd.DataFrame(history).to_csv(folder/'validation_history.csv',index=False)
            print('PAIR VALIDATION',seed,row,'best',best,flush=True)
    if best is None:raise ValueError('No validation check')
    model,opt,_,_=inspired.models('mlp',outer['X'].shape[1],Y.shape[1],seed)
    for epoch in range(best['epoch']):train_epoch(model,opt,outer,train)
    pred,counts=predict(model,outer,test)
    torch.save(model.state_dict(),folder/'model.pt')
    metadata=dict(best,inner_max_updates=core.PROTOCOL['epochs']*int(np.ceil(len(itr)/core.PROTOCOL['batch_size'])),final_updates=best['epoch']*int(np.ceil(len(train)/core.PROTOCOL['batch_size'])),epoch_policy='uniform_pairs_with_replacement_draw_count_equals_training_targets; inverse_target_pair_count_weights_normalized_globally',final_training_targets=len(train),final_training_pairs=int(np.isin(outer['target_index'],train).sum()))
    core.save_json(folder/'selection.json',metadata)
    return pred,counts,metadata


def verify_unit(unit,files):
    import sklearn,scipy
    info=json.loads((unit/'run.json').read_text());s=info['settings']
    expected=dict(input_components=64,input_space='pca',output_space='full',structure_layout='separate',learning_rate=1e-4,batch_size=128,context=256,hidden=64)
    if any(s.get(k)!=v for k,v in expected.items()) or s.get('source_input','present')!='present':raise ValueError('Reference protocol differs')
    current=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,scipy=scipy.__version__)
    if any(info['versions'][k]!=v for k,v in current.items()):raise ValueError('Use original reference environment')
    for path,checksum in info['input_sha256'].items():
        if path not in files:
            if core.digest(Path(path))!=checksum:raise ValueError('Reference input changed: '+path)
            files[path]=checksum
        elif files[path]!=checksum:raise ValueError('Reference input contracts differ')
    for path in [Path(m.__file__) for m in [core,pilot,inspired]]+[core.SNAPSHOT,Path(__file__).with_name('evaluate_repeated_predictions.py')]:
        matches=[v for n,v in info['source_sha256'].items() if Path(n).name==path.name]
        if matches!=[core.digest(path)]:raise ValueError('Shared reference source differs: '+path.name)
    return info


def summarize(a,ref,args,data,ids,meta,Y):
    from evaluate_repeated_predictions import score
    regression=[];classification=[]
    for seed in args['model_seeds']:
        for arm in ['basic','morgan']:
            seen=[];predictions=[]
            for fold in range(args['folds']):
                task=a.out/f'fold_{fold}'/f'seed_{seed}'/arm
                record=core.read_completion(task,dict(fold=fold,model_seed=seed,arm=arm,model='pair_mlp'))
                if record is None:raise ValueError('Incomplete pair result')
                with np.load(task/record['prediction_path']) as saved:seen.extend(saved['target_ids'].tolist());predictions.append(saved['prediction'])
            if len(set(seen))!=len(seen) or set(seen)!=set(ids):raise ValueError('OOF target mismatch')
            lookup=dict(zip(ids,range(len(ids))));index=np.array([lookup[i] for i in seen]);prediction=np.concatenate(predictions)
            reg,cls,_,_=score(Y[index],prediction,[.5,1.])
            regression.append(dict(condition='pair_'+arm,model_seed=seed,**reg))
            classification.extend([dict(condition='pair_'+arm,model_seed=seed,**r) for r in cls])
    old=pd.read_csv(ref/'pooled_by_model_seed.csv')
    all_reg=pd.concat([old,pd.DataFrame(regression)],ignore_index=True);all_reg.to_csv(a.out/'pooled_metrics.csv',index=False)
    old_cls=pd.read_csv(ref/'classification_by_model_seed.csv')
    all_cls=pd.concat([old_cls,pd.DataFrame(classification)],ignore_index=True);all_cls.to_csv(a.out/'classification_by_model_seed.csv',index=False)
    summary=all_reg.groupby('condition')[['rmse','mae','pearson_target_mean','spearman_target_mean','r2_probe_variance_weighted']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(a.out/'summary.csv')
    csummary=all_cls.groupby(['condition','threshold'])[['macro_f1','down_f1','up_f1','active_sign_accuracy']].agg(['mean','std']);csummary.columns=['_'.join(c) for c in csummary.columns];csummary.to_csv(a.out/'classification_summary.csv')
    wide=all_reg[all_reg.model_seed.ge(0)].pivot(index='model_seed',columns='condition',values='rmse');paired=[]
    for seed,row in wide.iterrows():
        for name,left,right in [('pair_minus_concat_basic','pair_basic','B_conditions_vitro'),('pair_minus_concat_morgan','pair_morgan','D_conditions_vitro_morgan'),('pair_basic_minus_target_conditions','pair_basic','A_conditions'),('pair_morgan_minus_target_and_structure','pair_morgan','C_conditions_morgan'),('morgan_minus_basic_in_pair_model','pair_morgan','pair_basic')]:paired.append(dict(model_seed=int(seed),comparison=name,rmse_difference=row[left]-row[right]))
    pd.DataFrame(paired).to_csv(a.out/'paired_rmse.csv',index=False)
    print('\nCOMPARISON\n'+summary.to_string(),flush=True)
    print('\nPAIRED DIFFERENCES\n'+pd.DataFrame(paired).to_string(index=False),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    # Default plan does not read server inputs.
    if not a.execute:
        print(json.dumps(dict(mode='plan_only',family='mlp',arms=['basic','morgan'],protocol='reuse completed reference folds/seeds/preprocessing; target-matched optimizer-step budget; target-equal weighted pair loss',default_reference_plan='5 folds x 3 model seeds x 2 structure arms = 30 conditions / up to 60 fits; actual count read from reference on execute'),indent=2));return
    ref=a.reference_run.resolve()
    if not json.loads((ref/'information_status.json').read_text()).get('all_complete'):raise ValueError('Completed four-input reference required')
    args=json.loads((ref/'run.json').read_text())['arguments']
    core.protect_output(a.out,[ref,Path(args['prepared']),Path(args['structure']),Path(args['smiles'])])
    sources=list(Path(__file__).parent.glob('*.py'))+[core.SNAPSHOT]
    reference_files=[ref/'run.json',ref/'pooled_by_model_seed.csv',ref/'classification_by_model_seed.csv',ref/'information_status.json']
    for seed in args['model_seeds']:
        for fold in range(args['folds']):
            unit=ref/'present'/f'fold_{fold}'/f'seed_{seed}'
            reference_files.extend(unit/n for n in ['run.json','compound_split.csv','baseline_predictions.npz'])
            for arm in ['basic','morgan']:
                reference_files.extend(unit/arm/n for n in ['inner_preprocess.joblib','final_preprocess.joblib','model_dimensions.json'])
    contract=dict(reference_run=str(ref),reference_sha256={str(f.resolve()):core.digest(f) for f in reference_files},source_sha256={str(f.resolve()):core.digest(f) for f in sources},training='per_epoch_uniform_pair_draws_equal_number_of_training_targets; global_inverse_pair_count_weights')
    core.contract_folder(a.out,contract)
    loader_args=SimpleNamespace(prepared=Path(args['prepared']),structure=Path(args['structure']))
    _,data,ids,meta,Y,_,_=core.load_data(loader_args)
    source=pd.read_csv(loader_args.prepared/'vitro_conditions.csv');source=source[source.condition_id.isin(np.unique(data.source_ids))].reset_index(drop=True)
    input_hashes={};old=core.PROTOCOL.copy();total=args['folds']*len(args['model_seeds'])*2;done=0
    torch.set_num_threads(args['threads'])
    from threadpoolctl import threadpool_limits
    try:
        with threadpool_limits(limits=args['threads']):
            for seed in args['model_seeds']:
                for fold in range(args['folds']):
                    unit=ref/'present'/f'fold_{fold}'/f'seed_{seed}';info=verify_unit(unit,input_hashes)
                    core.PROTOCOL.update(info['settings'])
                    split=pd.read_csv(unit/'compound_split.csv');test=np.flatnonzero(meta.compound.isin(split.loc[split.stage.eq('test'),'compound']));itr=np.flatnonzero(meta.compound.isin(split.loc[split.stage.eq('train'),'compound']));val=np.flatnonzero(meta.compound.isin(split.loc[split.stage.eq('validation'),'compound']));train=np.sort(np.r_[itr,val])
                    for arm in ['basic','morgan']:
                        folder=a.out/f'fold_{fold}'/f'seed_{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
                        job=dict(fold=fold,model_seed=seed,arm=arm,model='pair_mlp')
                        previous=core.read_completion(folder,job)
                        if previous is None:
                            inner_state=joblib.load(unit/arm/'inner_preprocess.joblib');outer_state=joblib.load(unit/arm/'final_preprocess.joblib')
                            inner=prepare(data,source,ids,meta,Y,inner_state,itr,arm);outer=prepare(data,source,ids,meta,Y,outer_state,train,arm)
                            pd.DataFrame(dict(source_id=outer['pair_source_ids'],target_id=outer['pair_target_ids'])).to_csv(folder/'pairs_used.csv',index=False)
                            np.savez_compressed(folder/'training_target_weights.npz',pair_target_rows=inner['target_index'],inner_weights=training_weights(inner['target_index'],itr)[1].numpy(),final_weights=training_weights(outer['target_index'],train)[1].numpy())
                            core.save_json(folder/'model_dimensions.json',dict(inner=pilot.model_parameter_report(inner['X'].shape[1],Y.shape[1]),final=pilot.model_parameter_report(outer['X'].shape[1],Y.shape[1]),reference_dimensions=json.loads((unit/arm/'model_dimensions.json').read_text())))
                            print('START',job,flush=True)
                            prediction,counts,selection=fit(inner,outer,Y,itr,val,train,test,seed,folder)
                            np.savez_compressed(folder/'predictions.npz',target_ids=ids[test],prediction=prediction,n_sources=counts)
                            core.save_json(folder/'complete.json',dict(job=job,prediction_path='predictions.npz',selection=selection,sha256={str(f.relative_to(folder)):core.digest(f) for f in folder.iterdir() if f.is_file()}))
                        with np.load(folder/'predictions.npz') as saved:
                            if not np.array_equal(saved['target_ids'],ids[test]):raise ValueError('Saved test target mismatch')
                            prediction=saved['prediction']
                        done+=1;core.save_json(a.out/'progress.json',dict(completed_jobs=done,total_jobs=total));print('COMPLETED',done,'/',total,'RMSE',core.rmse(Y[test],prediction),flush=True)
            summarize(a,ref,args,data,ids,meta,Y)
    finally:core.PROTOCOL.clear();core.PROTOCOL.update(old)
    core.save_json(a.out/'complete.json',dict(all_complete=True,jobs=total,source='existing reference input metadata and saved preprocessing',input_sha256=input_hashes,limitations='Different input width, pair objective and source averaging; not an isolated architecture or AIVIVE reproduction. Fixed folds and exploratory tests.'))

if __name__=='__main__':main()
