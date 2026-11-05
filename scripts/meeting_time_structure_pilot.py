"""Short meeting pilot: preserve observed source conditions, PCA, 3 structure arms x 3 models."""
import argparse
import json
from pathlib import Path
import platform
import time
import uuid
import bounded_descriptors
import repeated_latent_validation as core
import aivive_inspired_validation as inspired
import compare_structure_representations as chemistry
from evaluate_repeated_predictions import score


def schema_for(source):
    """Slots use input metadata only, never expression values or target outcomes."""
    counts=source.groupby(['compound','time_h','level','unit'],dropna=False).size()
    maximum=counts.groupby(level=['time_h','level','unit']).max()
    return [(float(t),str(level),str(unit),i) for (t,level,unit),n in sorted(maximum.items()) for i in range(int(n))]


def encoder_for(table,fit):
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    features=pd.DataFrame(dict(dose=np.log1p(table.dose.to_numpy(float)),time=table.time_h.to_numpy(float),
                               unit=table.unit.astype(str),level=table.level.astype(str)))
    enc=ColumnTransformer([('numeric',StandardScaler(),['dose','time']),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['unit','level'])])
    enc.fit(features.iloc[fit])
    return enc,enc.transform(features)


def prepare(data,source,slots,ids,meta,Y,fit,components=32):
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    from sklearn.decomposition import PCA
    compounds=set(meta.iloc[fit].compound)
    source_fit=np.flatnonzero(source.compound.isin(compounds))
    scaler=MinMaxScaler().fit(data.source[source.condition_id.to_numpy(int)[source_fit]])
    expression=scaler.transform(data.source[source.condition_id.to_numpy(int)])
    pca=PCA(n_components=min(components,len(source_fit)-1,expression.shape[1]),svd_solver='randomized',random_state=42).fit(expression[source_fit])
    keep=pca.singular_values_>pca.singular_values_[0]*1e-8
    if not keep.any():raise ValueError('No varying source PCA axes')
    values=pca.transform(expression)[:,keep]
    pscale=StandardScaler().fit(values[source_fit]);values=pscale.transform(values)
    senc,scond=encoder_for(source,source_fit)
    token=np.c_[values,scond,np.ones(len(source))].astype(np.float32)
    slot_index={slot:i for i,slot in enumerate(slots)}
    sequences={};sequence_audit=[]
    for compound,group in source.groupby('compound',sort=True):
        matrix=np.zeros((len(slots),token.shape[1]),dtype=np.float32)
        for (t,level,unit),g in group.groupby(['time_h','level','unit'],sort=True):
            g=g.sort_values(['dose','condition_id'])
            for occurrence,index in enumerate(g.index):
                matrix[slot_index[(float(t),str(level),str(unit),occurrence)]]=token[index]
                sequence_audit.append(dict(compound=compound,condition_id=int(source.loc[index,'condition_id']),
                    slot=slot_index[(float(t),str(level),str(unit),occurrence)]))
        sequences[compound]=matrix.ravel()
    tenc,target_conditions=encoder_for(meta,fit)
    X=np.c_[np.asarray([sequences[c] for c in meta.compound]),target_conditions].astype(np.float32)
    output=PCA(n_components=min(32,len(fit)-1,Y.shape[1]),svd_solver='randomized',random_state=42).fit(Y[fit])
    okeep=output.singular_values_>output.singular_values_[0]*1e-8
    z=output.transform(Y[fit])[:,okeep]
    scale=float(np.sqrt(np.mean(z*z)))
    if not okeep.any() or scale<=1e-8 or not np.isfinite(X).all():raise ValueError('Invalid PCA/inputs')
    Z=torch.zeros(len(Y),int(okeep.sum()));Z[fit]=torch.tensor(z/scale,dtype=torch.float32)
    state=dict(source_scaler=scaler,source_pca=pca,source_keep=keep,source_score_scaler=pscale,
        source_condition_encoder=senc,target_condition_encoder=tenc,slots=slots,source_slot_audit=sequence_audit,
        output_pca=output,output_keep=okeep,output_scale=scale,prediction_target='direct_response',
        source_feature_layout='slot_[PCA_scores,source_conditions,observed_flag]_then_target_conditions')
    return dict(X=torch.tensor(X),Z=Z,state=state)


def with_structure(base,data,meta,fit,raw,arm,descriptor_policy="bounded"):
    import numpy as np
    import torch
    width=max(data.structure.shape[1],len(raw.columns)+1)
    values=np.zeros((len(meta),width),dtype=np.float32)
    state=dict(base['state'],structure_representation=arm,structure_slots=width)
    if arm=='morgan':
        m=data.structure[[data.compound_to_index[c] for c in meta.compound]]
        values[:,:m.shape[1]-1]=m[:,:-1];values[:,-1]=m[:,-1]
    elif arm=='descriptor':
        if descriptor_policy=='bounded':
            ds=bounded_descriptors.fit(raw,meta.iloc[fit].compound)
            values,audit=bounded_descriptors.encode(raw,meta.compound.tolist(),ds,width)
            state.update(descriptor_preprocess=ds,descriptor_numeric_audit=audit)
        else:
            ds=chemistry.fit_descriptors(raw,meta.iloc[fit].compound)
            values,missing=chemistry.encode_descriptors(raw,meta.compound.tolist(),ds,width)
            state.update(descriptor_preprocess=ds,descriptor_missing_by_target=missing)
    return dict(X=torch.cat([base['X'],torch.tensor(values)],dim=1),Z=base['Z'],state=state)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','structure','smiles','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--model-seed',type=int,default=42)
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--arms',nargs='+',choices=['basic','morgan','descriptor'],default=['basic','morgan','descriptor'])
    p.add_argument('--families',nargs='+',choices=['mlp','cgan','ddpm'],default=['mlp','cgan','ddpm'])
    p.add_argument('--descriptor-policy',choices=['standard','bounded'],default='bounded')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--budget-minutes',type=float,default=60.)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if a.epochs<1 or a.threads<1 or a.budget_minutes<=0:p.error('Invalid epoch/thread/budget')
    if a.folds<2 or not 0<=a.fold<a.folds or any(s<0 or s>=2**32 for s in [a.split_seed,a.model_seed]):p.error('Invalid folds/seeds')
    if len(set(a.arms))!=len(a.arms) or len(set(a.families))!=len(a.families):p.error('Duplicate arms/families')
    n_jobs=len(a.arms)*len(a.families)
    epochs=sorted(set(e for e in [1,5,10,15,20]+list(range(40,a.epochs+1,20))+[a.epochs] if e<=a.epochs))
    settings=dict(inspired.SETTINGS,protocol_name='meeting_time_condition_pilot_v1',
        epochs=a.epochs,validation_epochs=epochs,batch_size=128,learning_rate=1e-4,
        input_components=32,output_components=32,steps=20,samples=4,
        input_transform='training_source_MinMax_then_PCA_then_StandardScaler',
        output_transform='training_response_PCA_shared_RMS',mlp_loss='PCA_score_MAE',
        cgan_loss='10_PCA_MAE_plus_2_BCE_no_cycle',pca=True,split_seed=a.split_seed,model_seed=a.model_seed,
        outer_folds=a.folds,evaluated_fold=a.fold,arms=a.arms,families=a.families,descriptor_policy=a.descriptor_policy,source_aggregation='none_fixed_metadata_slots_with_mask',
        slot_schema_policy='all_available_source_metadata_no_expression_or_target_outcomes')
    print(json.dumps(dict(mode='execute' if a.execute else 'plan_only',jobs=n_jobs,fits_up_to=2*n_jobs,
        settings=settings,soft_budget_minutes=a.budget_minutes,
        note='one exploratory compound holdout; no new job starts after budget, running job may finish later'),indent=2),flush=True)
    if not a.execute:return n_jobs
    started=time.monotonic()
    import numpy as np
    import pandas as pd
    import torch,sklearn,joblib,scipy
    from threadpoolctl import threadpool_limits
    files=[a.prepared/n for n in ['vitro_expression.npy','vivo_expression.npy','vitro_conditions.csv',
                                  'vivo_conditions.csv','pairs.csv.gz','probes.csv']]+[a.structure,a.smiles]
    core.protect_output(a.out,[a.prepared,a.structure,a.smiles])
    raw,caudit,rdversion=chemistry.calculate_descriptors(a.smiles)
    code=[Path(__file__),Path(core.__file__),Path(inspired.__file__),Path(chemistry.__file__),core.SNAPSHOT,
          Path(__file__).with_name('evaluate_repeated_predictions.py'),Path(bounded_descriptors.__file__)]
    contract=dict(settings=settings,arguments={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='execute'},
        input_sha256={str(f.resolve()):core.digest(f) for f in files},
        source_sha256={str(f.resolve()):core.digest(f) for f in code},
        versions=dict(python=platform.python_version(),rdkit=rdversion,numpy=np.__version__,pandas=pd.__version__,
                      torch=torch.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,scipy=scipy.__version__))
    core.contract_folder(a.out,contract)
    _,data,ids,meta,Y,ta,sa=core.load_data(a)
    pd.DataFrame(ta).to_csv(a.out/'target_control_audit.csv',index=False)
    pd.DataFrame(sa).to_csv(a.out/'source_control_audit.csv',index=False)
    raw.to_csv(a.out/'raw_descriptors.csv',index_label='compound');caudit.to_csv(a.out/'descriptor_calculation_audit.csv',index=False)
    coverage=data.structure_audit.copy().rename(columns={'structure_present':'morgan_present'})
    coverage['descriptor_present']=coverage.compound.map(raw.notna().any(axis=1)).fillna(False).astype(int)
    coverage.to_csv(a.out/'structure_coverage.csv',index=False)
    mapping=core.compound_folds(data.names,a.split_seed,a.folds);tn,vn=core.inner_compounds(mapping,a.fold,a.split_seed)
    itr=np.flatnonzero(meta.compound.isin(tn));val=np.flatnonzero(meta.compound.isin(vn))
    test=np.flatnonzero(meta.compound.map(mapping).eq(a.fold));train=np.sort(np.r_[itr,val])
    pd.DataFrame([dict(compound=c,stage='test' if f==a.fold else 'validation' if c in vn else 'train') for c,f in mapping.items()]).to_csv(a.out/'compound_split.csv',index=False)
    source=pd.read_csv(a.prepared/'vitro_conditions.csv')
    source=source[source.condition_id.isin(np.unique(data.source_ids))].reset_index(drop=True)
    if source[['compound','dose','unit','time_h','level']].isna().any().any():raise ValueError('Missing source metadata')
    slots=schema_for(source);core.save_json(a.out/'source_slots.json',slots)
    print('Compounds:',len(tn),len(vn),sum(f==a.fold for f in mapping.values()),'source slots:',len(slots),flush=True)
    torch.set_num_threads(a.threads)
    old=core.PROTOCOL.copy();hooks={k:getattr(core,k) for k in ['make_family_model','train_family_epoch']}
    results=[];classifications=[]
    def add_result(arm,kind,pred,selection=None):
        reg,cls,_,detail=score(Y[test],pred,[.5,1.])
        results.append(dict(arm=arm,model=kind,**reg,**(selection or {})))
        classifications.extend([dict(arm=arm,model=kind,**r) for r in cls])
        pd.DataFrame(results).to_csv(a.out/'results.csv',index=False)
        pd.DataFrame(classifications).to_csv(a.out/'classification.csv',index=False)
        return detail
    try:
        core.PROTOCOL.update(settings);core.make_family_model=inspired.models;core.train_family_epoch=inspired.train
        with threadpool_limits(limits=a.threads):
            inner=prepare(data,source,slots,ids,meta,Y,itr)
            outer=prepare(data,source,slots,ids,meta,Y,train)
            mean,counts=core.time_level_mean(meta,Y,train,test)
            core.save_json(a.out/'baseline_training_counts.json',counts)
            np.savez_compressed(a.out/'baseline_predictions.npz',target_ids=ids[test],time_level=mean)
            add_result('baseline','time_level_mean',mean);add_result('baseline','zero_response',np.zeros_like(mean))
            offset=np.zeros_like(Y)
            for arm in a.arms:
                ip=with_structure(inner,data,meta,itr,raw,arm,a.descriptor_policy);op=with_structure(outer,data,meta,train,raw,arm,a.descriptor_policy)
                armfolder=a.out/arm;armfolder.mkdir(exist_ok=True)
                joblib.dump(ip['state'],armfolder/'inner_preprocess.joblib');joblib.dump(op['state'],armfolder/'final_preprocess.joblib')
                if arm=='descriptor' and a.descriptor_policy=='bounded':
                    for name,pack in [('inner',ip),('final',op)]:
                        audit=pack['state']['descriptor_numeric_audit']
                        pd.DataFrame(dict(condition_id=ids,compound=meta.compound,
                            missing=audit['missing_selected_values'],clipped=audit['clipped_selected_values'])).to_csv(armfolder/(name+'_descriptor_numeric_audit.csv'),index=False)
                for kind in a.families:
                    job=dict(split_seed=a.split_seed,fold=a.fold,model_seed=a.model_seed,model=kind,arm=arm)
                    task=armfolder/kind;task.mkdir(exist_ok=True)
                    record=core.read_completion(task,job)
                    if record is None:
                        if time.monotonic()-started>a.budget_minutes*60:
                            core.save_json(a.out/'partial.json',dict(completed_jobs=len(results)-2,total_jobs=n_jobs,reason='soft_time_budget'))
                            print('SOFT BUDGET REACHED: partial results saved; no new training started.',flush=True)
                            return len(results)-2
                        attempt=task/('attempt_'+uuid.uuid4().hex);attempt.mkdir()
                        print('START',job,flush=True);t=time.monotonic()
                        pred,selection=core.fit_job(job,ip,op,Y,itr,val,test,train,offset,offset,attempt)
                        np.savez_compressed(attempt/'predictions.npz',target_ids=ids[test],prediction=pred)
                        core.save_json(attempt/'selection.json',dict(selection,elapsed_seconds=time.monotonic()-t))
                        record=dict(job=job,prediction_path=str((attempt/'predictions.npz').relative_to(task)),
                            selection=selection,sha256={str(f.relative_to(task)):core.digest(f) for f in attempt.iterdir() if f.is_file()})
                        core.save_json(task/'complete.json',record)
                    with np.load(task/record['prediction_path'],allow_pickle=False) as saved:
                        if not np.array_equal(saved['target_ids'],ids[test]):raise ValueError('Prediction target mismatch')
                        pred=saved['prediction']
                    detail=add_result(arm,kind,pred,record['selection'])
                    pd.DataFrame(dict(condition_id=ids[test],compound=meta.iloc[test].compound.to_numpy(),**detail)).to_csv(task/'condition_metrics.csv',index=False)
                    print(f'COMPLETED {len(results)-2}/{n_jobs}; elapsed {(time.monotonic()-started)/60:.1f} min; RMSE={results[-1]["rmse"]:.6f}',flush=True)
            core.save_json(a.out/'completed.json',dict(jobs=n_jobs,scope='single_holdout_exploratory_not_full_5fold'))
            if (a.out/'partial.json').exists():(a.out/'partial.json').unlink()
            print(pd.DataFrame(results)[['arm','model','rmse','mae','pearson_target_mean']].to_string(index=False),flush=True)
            print('ALL COMPLETE:',a.out,flush=True)
    finally:
        core.PROTOCOL.clear();core.PROTOCOL.update(old)
        for k,v in hooks.items():setattr(core,k,v)
    return n_jobs


if __name__=='__main__':main()
