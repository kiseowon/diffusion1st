"""First-meeting experiment: identical MLP, with/without own Morgan features."""
import argparse
import copy
import json
from pathlib import Path
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from threadpoolctl import threadpool_limits


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def condition_features(pairs):
    result={}
    for suffix in ['source','target']:
        result[f'{suffix}_dose']=np.log1p(pairs[f'dose_{suffix}'].to_numpy(float))
        result[f'{suffix}_time']=pairs[f'time_h_{suffix}'].to_numpy(float)
        for key in ['unit','level']:result[f'{suffix}_{key}']=pairs[f'{key}_{suffix}'].astype(str).to_numpy()
    return pd.DataFrame(result)


class Data:
    def __init__(self,a):
        self.pairs=pd.read_csv(a.prepared/'pairs.csv.gz')
        self.source=np.load(a.prepared/'vitro_expression.npy')
        self.target=np.load(a.prepared/'vivo_expression.npy')
        self.vo=pd.read_csv(a.prepared/'vivo_conditions.csv')
        probes=pd.read_csv(a.prepared/'probes.csv')['probe']
        if probes.duplicated().any() or self.source.shape[1]!=len(probes) or self.target.shape[1]!=len(probes):raise ValueError('Probe dimension mismatch')
        for name,matrix in [('vitro',self.source),('vivo',self.target)]:
            table=pd.read_csv(a.prepared/f'{name}_conditions.csv')
            if not np.array_equal(table.condition_id,np.arange(len(matrix))) or not np.isfinite(matrix).all():raise ValueError('Invalid condition-expression alignment')
            suffix='source' if name=='vitro' else 'target'
            ids=self.pairs[f'condition_id_{suffix}'].to_numpy(int)
            if (ids<0).any() or (ids>=len(matrix)).any():raise ValueError('Condition ID range error')
            if not np.array_equal(table.iloc[ids].compound.to_numpy(),self.pairs.compound.to_numpy()):raise ValueError('Compound mismatch')
        if not self.pairs.level_target.eq('Control').any():raise ValueError('Use prepared_controls including target controls')
        if self.pairs.duplicated(['condition_id_source','condition_id_target']).any():raise ValueError('Duplicate source-target pair')
        self.features=condition_features(self.pairs)
        st=pd.read_csv(a.structure);st.compound=st.compound.astype(str).str.strip()
        if st.compound.duplicated().any():raise ValueError('Duplicate structure names')
        st=st.set_index('compound')
        if not np.isfinite(st.to_numpy(float)).all() or not np.isin(st.to_numpy(),[0,1]).all():raise ValueError('Binary Morgan fingerprint required')
        st=st.copy();st['structure_present']=1.
        self.names=sorted(self.pairs.compound.unique())
        self.structure=st.reindex(self.names).fillna(0).to_numpy(np.float32)
        self.compound_to_index={n:i for i,n in enumerate(self.names)}
        self.compound_indices=self.pairs.compound.map(self.compound_to_index).to_numpy(int)
        self.structure_audit=pd.DataFrame({'compound':self.names,'structure_present':self.structure[:,-1].astype(int)})
        self.source_ids=self.pairs.condition_id_source.to_numpy(int)
        self.target_ids=self.pairs.condition_id_target.to_numpy(int)


def prepare(data,tr,ev):
    unique_source=np.unique(data.source_ids[tr]);unique_target=np.unique(data.target_ids[tr])
    source_scale=StandardScaler().fit(data.source[unique_source])
    source=source_scale.transform(data.source).astype(np.float32)
    numeric=[c for c in data.features if c.endswith(('_dose','_time'))]
    category=[c for c in data.features if c not in numeric]
    encoder=ColumnTransformer([('numeric',StandardScaler(),numeric),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),category)])
    encoder.fit(data.features.iloc[tr])
    conditions=encoder.transform(data.features).astype(np.float32)
    # Centering only: retain original probe-space MSE and output all probes directly.
    target_mean=data.target[unique_target].mean(0).astype(np.float32)
    _,inverse,counts=np.unique(data.target_ids[tr],return_inverse=True,return_counts=True)
    weights=(1/counts[inverse]).astype(np.float32);weights=weights/weights.mean()
    return dict(source=torch.tensor(source),conditions=torch.tensor(conditions),structure=torch.tensor(data.structure),
        target=torch.tensor(data.target-target_mean),target_mean=target_mean,weights=torch.tensor(weights),
        preprocess=dict(source_scale=source_scale,condition_encoder=encoder,target_mean=target_mean))


class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,output_dim))
    def forward(self,source,condition,structure):
        return self.net(torch.cat([source,condition,structure],dim=1))


def inputs(data,pack,rows,use_structure):
    source=pack['source'][data.source_ids[rows]]
    condition=pack['conditions'][rows]
    structure=pack['structure'][data.compound_indices[rows]]
    if not use_structure:structure=torch.zeros_like(structure)
    return source,condition,structure


def model_for(data,pack,seed):
    seed_all(seed)
    return MLP(pack['source'].shape[1]+pack['conditions'].shape[1]+pack['structure'].shape[1],data.target.shape[1])


def train_epoch(model,opt,data,pack,tr,use_structure,batch_size):
    model.train();total=0.
    for local in torch.randperm(len(tr)).split(batch_size):
        rows=tr[local.numpy()]
        pred=model(*inputs(data,pack,rows,use_structure))
        loss=((pred-pack['target'][data.target_ids[rows]]).square().mean(1)*pack['weights'][local]).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        total+=loss.item()*len(rows)
    return total/len(tr)


@torch.no_grad()
def predict(model,data,pack,rows,use_structure,batch_size):
    model.eval();ids=np.unique(data.target_ids[rows]);lookup={int(cid):i for i,cid in enumerate(ids)}
    sums=np.zeros((len(ids),data.target.shape[1]),np.float64);counts=np.zeros(len(ids),int);pair_sse=0.
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]
        pred=model(*inputs(data,pack,batch,use_structure)).numpy()+pack['target_mean']
        if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
        target_ids=data.target_ids[batch];positions=np.array([lookup[int(i)] for i in target_ids])
        np.add.at(sums,positions,pred);np.add.at(counts,positions,1)
        pair_sse+=float(np.sum((pred.astype(float)-data.target[target_ids])**2))
    return ids,(sums/counts[:,None]).astype(np.float32),float(np.sqrt(pair_sse/(len(rows)*data.target.shape[1])))


def evaluate(data,ids,pred):
    truth=data.target[ids];table=data.vo
    absolute=float(np.sqrt(np.mean((pred.astype(float)-truth)**2)))
    controls=table.loc[table.level.eq('Control')]
    if controls.duplicated(['compound','unit','time_h']).any():raise ValueError('Ambiguous control conditions')
    control_map=controls.set_index(['compound','unit','time_h']).condition_id.to_dict()
    lookup={int(cid):i for i,cid in enumerate(ids)};rows=[]
    for i,cid in enumerate(ids):
        row=table.iloc[cid]
        if row.level=='Control':continue
        control=control_map.get((row.compound,row.unit,row.time_h))
        if control is None or control not in lookup:raise ValueError('Predicted matched control missing')
        observed=data.target[cid]-data.target[control]
        predicted=pred[i]-pred[lookup[control]]
        rows.append(dict(compound=row.compound,condition_id=int(cid),time_h=row.time_h,level=row.level,
            response_mse=float(np.mean((predicted.astype(float)-observed)**2)),zero_response_mse=float(np.mean(observed.astype(float)**2))))
    detail=pd.DataFrame(rows)
    if detail.empty:raise ValueError('No treated target conditions')
    response=float(np.sqrt(detail.response_mse.mean()));zero=float(np.sqrt(detail.zero_response_mse.mean()))
    return dict(absolute_rmse=absolute,response_rmse=response,zero_response_rmse=zero,ratio_to_no_change=response/zero if zero>0 else np.nan,n_target_conditions=len(ids),n_treated_conditions=len(detail)),detail


def run(a,model_factory=None,prepare_fn=None,variants=None):
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('Use a fresh output directory')
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(a.threads)
    model_factory=model_for if model_factory is None else model_factory
    prepare_fn=prepare if prepare_fn is None else prepare_fn
    data=Data(a);data.structure_audit.to_csv(a.out/'structure_audit.csv',index=False)
    names=np.random.default_rng(a.seed).permutation(data.names)
    n_test=max(1,int(np.ceil(.2*len(names))));test_names=set(names[:n_test])
    remaining=names[n_test:];n_val=max(1,int(np.ceil(.2*len(remaining))))
    val_names=set(remaining[:n_val]);train_names=set(remaining[n_val:])
    if len(train_names)<2:raise ValueError('Too few compounds for 3-way split')
    sets=[train_names,val_names,test_names]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):raise ValueError('Compound split overlap')
    inner_tr=np.flatnonzero(data.pairs.compound.isin(train_names));inner_val=np.flatnonzero(data.pairs.compound.isin(val_names))
    outer_tr=np.flatnonzero(~data.pairs.compound.isin(test_names));outer_te=np.flatnonzero(data.pairs.compound.isin(test_names))
    split=[]
    for stage,values in zip(['inner_train','inner_validation','outer_test'],sets):split.extend(dict(compound=n,stage=stage) for n in sorted(values))
    pd.DataFrame(split).to_csv(a.out/'compound_split.csv',index=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(protocol='same MLP with own Morgan features versus zero-filled structure slots',selection='internal target-averaged absolute MSE',input_policy='each available treated vitro condition predicts each target; average predictions per target for main evaluation',target='condition-mean expression, all probes directly; no target PCA')
    (a.out/'config.json').write_text(json.dumps(config,indent=2))
    print(f'Compounds train/val/test: {list(map(len,sets))}; probes={data.target.shape[1]}',flush=True)
    print('Missing structure compounds:',data.structure_audit.loc[data.structure_audit.structure_present.eq(0),'compound'].tolist(),flush=True)
    inner=prepare_fn(data,inner_tr,inner_val);selected=[];history=[]
    for name,use_structure in (variants if variants is not None else [('expression_only',False),('expression_structure',True)]):
        model=model_factory(data,inner,a.seed)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        best=float('inf');best_epoch=None;best_state=None
        for epoch in range(1,a.epochs+1):
            loss=train_epoch(model,opt,data,inner,inner_tr,use_structure,a.batch_size)
            if epoch==1 or epoch%a.check_every==0 or epoch==a.epochs:
                ids,pred,_=predict(model,data,inner,inner_val,use_structure,a.batch_size)
                mse=float(np.mean((pred.astype(float)-data.target[ids])**2))
                if mse<best:best=mse;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
                history.append(dict(model=name,epoch=epoch,train_loss=loss,validation_rmse=np.sqrt(mse),best_epoch=best_epoch))
                pd.DataFrame(history).to_csv(a.out/'validation_history.csv',index=False)
                print(f'{name} epoch={epoch}/{a.epochs}, val_RMSE={np.sqrt(mse):.5f}, best_epoch={best_epoch}',flush=True)
        selected.append((name,use_structure,best_epoch,best))
        torch.save(best_state,a.out/f'{name}_inner_best.pt')
    del inner
    outer=prepare_fn(data,outer_tr,outer_te);joblib.dump(outer['preprocess'],a.out/'preprocess.joblib')
    results=[];response_details=[]
    for name,use_structure,epochs,best in selected:
        model=model_factory(data,outer,a.seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        print(f'{name}: outer refit {epochs} epochs',flush=True)
        for epoch in range(1,epochs+1):
            loss=train_epoch(model,opt,data,outer,outer_tr,use_structure,a.batch_size)
            if epoch==1 or epoch%10==0 or epoch==epochs:print(f'{name} refit={epoch}/{epochs}',flush=True)
        ids,pred,pair_rmse=predict(model,data,outer,outer_te,use_structure,a.batch_size)
        metrics,detail=evaluate(data,ids,pred)
        metrics.update(model=name,selected_epoch=epochs,validation_rmse=np.sqrt(best),pair_rmse=pair_rmse,parameters=sum(p.numel() for p in model.parameters()))
        detail['model']=name;response_details.append(detail);results.append(metrics)
        np.savez_compressed(a.out/f'{name}_predictions.npz',target_ids=ids,prediction=pred)
        torch.save(model.state_dict(),a.out/f'{name}.pt')
        print(f'{name}: TEST absolute={metrics["absolute_rmse"]:.5f}, response={metrics["response_rmse"]:.5f}',flush=True)
    summary=pd.DataFrame(results);summary.to_csv(a.out/'comparison.csv',index=False)
    pd.concat(response_details,ignore_index=True).to_csv(a.out/'response_condition_metrics.csv',index=False)
    print(summary.to_string(index=False));print('Saved:',a.out,flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--structure',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--check-every',type=int,default=1);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if min(a.epochs,a.check_every,a.batch_size,a.threads)<1:p.error('Positive settings required')
    with threadpool_limits(limits=a.threads):run(a)
