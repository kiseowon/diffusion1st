"""Single-condition exploratory IVIVE toy: 24h High -> 24h High, nine model/structure conditions."""
import argparse
import json
import platform
from pathlib import Path
import time
import uuid
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
import repeated_latent_validation as core
import aivive_inspired_validation as inspired
import meeting_time_structure_pilot as pilot
import compare_structure_representations as chemistry
import bounded_descriptors
from evaluate_repeated_predictions import score

MODEL_SETTINGS=dict(expression_embedding=64,structure_embedding=32,context=64,hidden=128)


def cohort(data,meta,Y,source,raw,structure_audit):
    sources=source[source.time_h.eq(24)&source.level.eq('High')]
    targets=meta[meta.time_h.eq(24)&meta.level.eq('High')]
    parsed=structure_audit.set_index('compound').parsed.to_dict()
    rows=[];audit=[];sids=[];target_rows=[]
    for name in sorted(data.names):
        s=sources[sources.compound.eq(name)];t=targets[targets.compound.eq(name)]
        morgan=bool(data.structure[data.compound_to_index[name],-1])
        descriptor=bool(parsed.get(name,False)) and name in raw.index and bool(raw.loc[name].notna().any())
        reasons=[]
        if len(s)!=1:reasons.append('source_24h_High_count_'+str(len(s)))
        if len(t)!=1:reasons.append('target_24h_High_count_'+str(len(t)))
        if not morgan:reasons.append('missing_Morgan')
        if not descriptor:reasons.append('missing_parsed_descriptor_structure')
        audit.append(dict(compound=name,source_count=len(s),target_count=len(t),morgan_available=morgan,descriptor_available=descriptor,included=not reasons,exclusion_reason=';'.join(reasons)))
        if reasons:continue
        sr=s.iloc[0];tr=t.iloc[0]
        sids.append(int(sr.condition_id));target_rows.append(int(t.index[0]))
        rows.append(dict(compound=name,source_condition_id=int(sr.condition_id),target_condition_id=int(tr.condition_id),source_time_h=float(sr.time_h),target_time_h=float(tr.time_h),source_level=str(sr.level),target_level=str(tr.level),source_dose=float(sr.dose),source_unit=str(sr.unit),target_dose=float(tr.dose),target_unit=str(tr.unit)))
    if len(rows)<15:raise ValueError('Fewer than 15 common eligible compounds; inspect available 24h High observations and structures')
    return pd.DataFrame(rows),np.asarray(sids),Y[np.asarray(target_rows)],pd.DataFrame(audit)


def prepare(data,table,sids,Y,raw,fit,components):
    if components>min(len(fit)-1,data.source.shape[1]):raise ValueError('Too few training compounds/probes for requested PCA dimension')
    scaler=MinMaxScaler().fit(data.source[sids[fit]])
    source=scaler.transform(data.source[sids])
    pca=PCA(n_components=components,svd_solver='randomized',random_state=42).fit(source[fit])
    keep=pca.singular_values_>pca.singular_values_[0]*1e-8
    if int(keep.sum())!=components:raise ValueError('Requested PCA exceeds nondegenerate rank; choose dimension using training data only')
    e=pca.transform(source);pscale=StandardScaler().fit(e[fit]);e=pscale.transform(e).astype(np.float32)
    mean=Y[fit].mean(0);centered=Y[fit]-mean;scale=float(np.sqrt(np.mean(centered**2)))
    if not np.isfinite(scale) or scale<=1e-8:raise ValueError('Invalid training target RMS')
    Z=torch.zeros(Y.shape,dtype=torch.float32);Z[fit]=torch.tensor(centered/scale,dtype=torch.float32)
    width=max(data.structure.shape[1]-1,len(raw.columns))
    names=table.compound.tolist();m=data.structure[[data.compound_to_index[c] for c in names],:-1]
    morgan=np.zeros((len(table),width),np.float32);morgan[:,:m.shape[1]]=m
    descriptor=bounded_descriptors.fit(raw,table.iloc[fit].compound)
    encoded,audit=bounded_descriptors.encode(raw,names,descriptor,width+1)
    # Common complete-structure cohort: no representation-specific presence flags.
    states=dict(source_scaler=scaler,source_pca=pca,source_score_scaler=pscale,output_space='full',output_mean=mean,output_scale=scale,prediction_target='direct_response',source_expression_dimension=components,source_explained_variance=float(pca.explained_variance_ratio_.sum()),target_roundtrip_max_abs_error=float(np.max(np.abs((centered/scale)*scale+mean-Y[fit]))),descriptor_preprocess=descriptor,descriptor_numeric_audit=audit,structure_width=width,structure_presence_flags=False,source_condition='24h_High',target_condition='24h_High',dose_metadata_not_input=True)
    packs={}
    for arm,structure in [('basic',np.zeros_like(morgan)),('morgan',morgan),('descriptor',encoded[:,:-1])]:
        X=np.c_[e,structure].astype(np.float32)
        if not np.isfinite(X).all():raise ValueError('Nonfinite toy features')
        packs[arm]=dict(X=torch.tensor(X),Z=Z,state=dict(states,structure_representation=arm))
    return packs


class Context(nn.Module):
    def __init__(self,input_dim):
        super().__init__();d=core.PROTOCOL['input_components'];self.d=d
        self.expression=nn.Sequential(nn.Linear(d,64),nn.SiLU())
        self.structure=nn.Sequential(nn.Linear(input_dim-d,32),nn.SiLU())
        self.fuse=nn.Sequential(nn.Linear(96,64),nn.SiLU())
    def forward(self,x):return self.fuse(torch.cat([self.expression(x[:,:self.d]),self.structure(x[:,self.d:])],dim=1))


def models(kind,input_dim,output_dim,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__();self.context=Context(input_dim);self.latent_dim=output_dim
            extra=0 if kind=='mlp' else core.PROTOCOL['cgan_noise_dim'] if kind=='cgan' else output_dim+16
            self.head=nn.Sequential(nn.Linear(64+extra,128),nn.SiLU(),nn.Linear(128,output_dim))
            if kind=='ddpm':self.time=nn.Embedding(core.PROTOCOL['steps'],16)
        def forward(self,x,noisy,t=None):
            context=self.context(x)
            if kind=='mlp':return self.head(context)
            extra=noisy if kind=='cgan' else torch.cat([noisy,self.time(t)],dim=1)
            return self.head(torch.cat([context,extra],dim=1))
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__();self.context=Context(input_dim)
            self.head=nn.Sequential(nn.Linear(64+output_dim,128),nn.SiLU(),nn.Linear(128,1))
        def forward(self,x,y):return self.head(torch.cat([self.context(x),y],dim=1)).squeeze(1)
    model=Predictor();opt=torch.optim.Adam(model.parameters(),lr=core.PROTOCOL['learning_rate'])
    if kind!='cgan':return model,opt,None,None
    disc=Discriminator();return model,opt,disc,torch.optim.SGD(disc.parameters(),lr=1e-4,momentum=.9)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','structure','smiles','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--input-components',type=int,default=64);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--split-seed',type=int,default=42);p.add_argument('--model-seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--fold',type=int,default=0);p.add_argument('--folds',type=int,default=5);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    if min(a.input_components,a.epochs,a.threads)<1 or a.folds<2 or not 0<=a.fold<a.folds:p.error('Invalid dimensions/epochs/folds')
    if len(set(a.model_seeds))!=len(a.model_seeds) or any(s<0 or s>=2**32 for s in [a.split_seed]+a.model_seeds):p.error('Invalid seeds')
    checks=sorted({e for e in [1,5,10,15,20]+list(range(40,a.epochs+1,20))+[a.epochs] if e<=a.epochs})
    settings=dict(protocol='fixed_24h_High_toy_v1',input_components=a.input_components,epochs=a.epochs,validation_epochs=checks,batch_size=32,learning_rate=1e-4,weight_decay=0.,gradient_clip=1.,steps=20,samples=4,cgan_noise_dim=16,cache_diffusion_context=True,prediction='x0',schedule='cosine',output='all_probes_no_output_PCA',input='one_source_condition_no_time_dose_metadata',models=MODEL_SETTINGS,losses=dict(mlp='MAE',cgan='10_MAE_plus_2_BCE',ddpm='x0_MSE'),purpose='single_holdout_exploratory; model seeds vary initialization and generation randomness, not independent compound splits')
    jobs=9*len(a.model_seeds)
    print(json.dumps(dict(mode='execute' if a.execute else 'plan_only',jobs=jobs,fits_up_to=2*jobs,settings=settings,cohort='one source24hHigh + one target24hHigh + matched controls + Morgan + parsed descriptor structure; identical cohort across all arms'),indent=2),flush=True)
    if not a.execute:return jobs
    import sklearn,scipy
    from threadpoolctl import threadpool_limits
    core.protect_output(a.out,[a.prepared,a.structure,a.smiles])
    files=[a.prepared/n for n in ['vitro_expression.npy','vivo_expression.npy','vitro_conditions.csv','vivo_conditions.csv','pairs.csv.gz','probes.csv']]+[a.structure,a.smiles]
    sources=list(Path(__file__).parent.glob('*.py'))+[core.SNAPSHOT]
    raw,structure_audit,rdversion=chemistry.calculate_descriptors(a.smiles)
    contract=dict(arguments={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='execute'},settings=settings,input_sha256={str(f.resolve()):core.digest(f) for f in files},source_sha256={str(f.resolve()):core.digest(f) for f in sources},versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,joblib=joblib.__version__,rdkit=rdversion))
    core.contract_folder(a.out,contract)
    _,data,ids,meta,Y,ta,sa=core.load_data(a)
    source=pd.read_csv(a.prepared/'vitro_conditions.csv');source=source[source.condition_id.isin(np.unique(data.source_ids))].reset_index(drop=True)
    table,sids,Y,audit=cohort(data,meta,Y,source,raw,structure_audit)
    audit.to_csv(a.out/'cohort_audit.csv',index=False);table.to_csv(a.out/'cohort.csv',index=False)
    raw.to_csv(a.out/'raw_descriptors.csv',index_label='compound');structure_audit.to_csv(a.out/'descriptor_calculation_audit.csv',index=False)
    pd.DataFrame(ta).to_csv(a.out/'target_control_audit.csv',index=False);pd.DataFrame(sa).to_csv(a.out/'source_control_audit.csv',index=False)
    mapping=core.compound_folds(table.compound,a.split_seed,a.folds);tn,vn=core.inner_compounds(mapping,a.fold,a.split_seed)
    itr=np.flatnonzero(table.compound.isin(tn));val=np.flatnonzero(table.compound.isin(vn));test=np.flatnonzero(table.compound.map(mapping).eq(a.fold));train=np.sort(np.r_[itr,val])
    assignments=table.copy();assignments['stage']=np.where(table.compound.map(mapping).eq(a.fold),'test',np.where(table.compound.isin(vn),'validation','train'));assignments.to_csv(a.out/'compound_split.csv',index=False)
    print('COMMON COHORT:',len(table),'compounds; train/validation/test:',len(itr),len(val),len(test),flush=True)
    torch.set_num_threads(a.threads);old=core.PROTOCOL.copy();hooks={k:getattr(core,k) for k in ['make_family_model','train_family_epoch','restore']};regression=[];classification=[]
    def evaluate(arm,kind,seed,pred,selection=None):
        reg,cls,_,detail=score(Y[test],pred,[.5,1.])
        regression.append(dict(arm=arm,model=kind,model_seed=seed,**reg,**(selection or {})))
        classification.extend([dict(arm=arm,model=kind,model_seed=seed,**r) for r in cls])
        pd.DataFrame(regression).to_csv(a.out/'by_seed.csv',index=False);pd.DataFrame(classification).to_csv(a.out/'classification_by_seed.csv',index=False)
        return detail
    try:
        core.PROTOCOL.update(settings);core.make_family_model=models;core.train_family_epoch=inspired.train;core.restore=pilot.restore_output
        with threadpool_limits(limits=a.threads):
            inner=prepare(data,table,sids,Y,raw,itr,a.input_components);outer=prepare(data,table,sids,Y,raw,train,a.input_components)
            core.save_json(a.out/'preprocessing_summary.json',dict(inner_input_components=inner['basic']['state']['source_expression_dimension'],final_input_components=outer['basic']['state']['source_expression_dimension'],inner_explained_variance=inner['basic']['state']['source_explained_variance'],final_explained_variance=outer['basic']['state']['source_explained_variance'],inner_target_scale=inner['basic']['state']['output_scale'],final_target_scale=outer['basic']['state']['output_scale'],output_probes=Y.shape[1],no_target_dimension_reduction=True))
            mean=np.repeat(Y[train].mean(0)[None,:],len(test),axis=0)
            np.savez_compressed(a.out/'baseline_predictions.npz',target_ids=table.iloc[test].target_condition_id.to_numpy(),training_mean=mean,zero=np.zeros_like(mean))
            evaluate('baseline','training_compound_mean',-1,mean);evaluate('baseline','zero_response',-1,np.zeros_like(mean))
            completed=0
            for arm in ['basic','morgan','descriptor']:
                folder=a.out/arm;folder.mkdir(exist_ok=True)
                for label,packs in [('inner',inner),('final',outer)]:
                    joblib.dump(packs[arm]['state'],folder/f'{label}_preprocess.joblib')
                    if arm=='descriptor':
                        numeric=packs[arm]['state']['descriptor_numeric_audit'];pd.DataFrame(dict(compound=table.compound,missing=numeric['missing_selected_values'],clipped=numeric['clipped_selected_values'])).to_csv(folder/f'{label}_descriptor_numeric_audit.csv',index=False)
                for seed in a.model_seeds:
                    for kind in ['mlp','cgan','ddpm']:
                        task=folder/f'seed_{seed}'/kind;task.mkdir(parents=True,exist_ok=True);job=dict(model=kind,model_seed=seed,arm=arm,split_seed=a.split_seed,fold=a.fold)
                        record=core.read_completion(task,job)
                        if record is None:
                            attempt=task/('attempt_'+uuid.uuid4().hex);attempt.mkdir();started=time.monotonic()
                            m,_,d,_=models(kind,inner[arm]['X'].shape[1],Y.shape[1],seed)
                            core.save_json(attempt/'model_dimensions.json',dict(input_dimension=inner[arm]['X'].shape[1],output_dimension=Y.shape[1],model_parameters=sum(p.numel() for p in m.parameters()),discriminator_parameters=0 if d is None else sum(p.numel() for p in d.parameters())))
                            del m,d
                            print('START',job,flush=True)
                            offset=np.zeros_like(Y)
                            pred,selection=core.fit_job(job,inner[arm],outer[arm],Y,itr,val,test,train,offset,offset,attempt)
                            np.savez_compressed(attempt/'predictions.npz',target_ids=table.iloc[test].target_condition_id.to_numpy(),prediction=pred)
                            record=dict(job=job,prediction_path=str((attempt/'predictions.npz').relative_to(task)),selection=dict(selection,elapsed_seconds=time.monotonic()-started),sha256={str(f.relative_to(task)):core.digest(f) for f in attempt.iterdir() if f.is_file()})
                            core.save_json(task/'complete.json',record)
                        with np.load(task/record['prediction_path']) as saved:
                            if not np.array_equal(saved['target_ids'],table.iloc[test].target_condition_id):raise ValueError('Target mismatch')
                            pred=saved['prediction']
                        detail=evaluate(arm,kind,seed,pred,record['selection']);pd.DataFrame(dict(compound=table.iloc[test].compound.to_numpy(),target_id=table.iloc[test].target_condition_id.to_numpy(),**detail)).to_csv(task/'condition_metrics.csv',index=False)
                        completed+=1;core.save_json(a.out/'progress.json',dict(completed=completed,total=jobs));print('COMPLETE',completed,'/',jobs,flush=True)
    finally:
        core.PROTOCOL.clear();core.PROTOCOL.update(old)
        for name,hook in hooks.items():setattr(core,name,hook)
    reg=pd.DataFrame(regression);cls=pd.DataFrame(classification)
    summary=reg.groupby(['arm','model'])[['rmse','mae','pearson_target_mean','spearman_target_mean','r2_probe_variance_weighted']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(a.out/'summary.csv')
    cs=cls.groupby(['arm','model','threshold'])[['macro_f1','down_f1','up_f1','active_sign_accuracy','actual_active_fraction','predicted_active_fraction']].agg(['mean','std']);cs.columns=['_'.join(c) for c in cs.columns];cs.to_csv(a.out/'classification_summary.csv')
    paired=[];wide=reg[reg.model_seed.ge(0)].pivot(index='model_seed',columns=['arm','model'],values='rmse')
    for seed,row in wide.iterrows():
        for kind in ['mlp','cgan','ddpm']:
            for left,right in [('morgan','basic'),('descriptor','basic'),('descriptor','morgan')]:paired.append(dict(model_seed=int(seed),comparison=f'{left}_minus_{right}_{kind}',rmse_difference=row[left,kind]-row[right,kind]))
        for arm in ['basic','morgan','descriptor']:
            for left,right in [('ddpm','mlp'),('ddpm','cgan')]:paired.append(dict(model_seed=int(seed),comparison=f'{arm}_{left}_minus_{right}',rmse_difference=row[arm,left]-row[arm,right]))
    baselines=reg[reg.model_seed.eq(-1)].set_index('model').rmse.to_dict()
    for seed,row in wide.iterrows():
        for arm in ['basic','morgan','descriptor']:
            for kind in ['mlp','cgan','ddpm']:
                for baseline,value in baselines.items():paired.append(dict(model_seed=int(seed),comparison=f'{arm}_{kind}_minus_{baseline}',rmse_difference=row[arm,kind]-value))
    pd.DataFrame(paired).to_csv(a.out/'paired_rmse.csv',index=False)
    core.save_json(a.out/'complete.json',dict(all_complete=True,jobs=jobs,cohort_compounds=len(table),test_compounds=len(test),evaluated_fold=a.fold,folds_defined=a.folds,all_folds_evaluated=False,scope='single compound holdout with model initialization repeats, exploratory; High doses differ across compounds'))
    print('\nTOY RESULTS\n'+summary.to_string(),flush=True)
    print('COMPLETE:',a.out,flush=True)
    return jobs

if __name__=='__main__':main()
