"""Four-way MLP ablation: expression, own structure, pathway summary, both."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import first_structure_mlp as first


def load_projection(annotation,prepared):
    annotation=Path(annotation)
    provenance=json.loads((annotation/'provenance.json').read_text())
    probes=pd.read_csv(Path(prepared)/'probes.csv')['probe'].astype(str).tolist()
    if provenance['species']!='Rattus norvegicus' or provenance['probe_order']!=probes:raise ValueError('Species/probe order mismatch')
    projection=np.load(annotation/'projection.npy').astype(np.float64)
    pathways=pd.read_csv(annotation/'pathway_audit.csv')
    if projection.shape!=(len(probes),len(pathways)) or len(pathways)==0 or pathways.pathway.duplicated().any():raise ValueError('Pathway layout mismatch')
    if not np.isfinite(projection).all() or (projection<0).any():raise ValueError('Invalid pathway weights')
    np.testing.assert_allclose(projection.sum(0),1.,rtol=0,atol=1e-6)
    return projection,pathways,provenance


def prepare_pathways(data,tr,ev,projection,pathways,provenance):
    pack=first.prepare(data,tr,ev)
    # Gene standardization was fitted to unique TRAINING in vitro source conditions.
    gene_expression=pack['source'].numpy()
    scores=(gene_expression.astype(np.float64)@projection).astype(np.float32)
    unique_sources=np.unique(data.source_ids[tr])
    pathway_scale=StandardScaler().fit(scores[unique_sources])
    normalized=pathway_scale.transform(scores).astype(np.float32)
    pack['source']=torch.cat([pack['source'],torch.tensor(normalized)],dim=1)
    pack['preprocess'].update(pathway_scale=pathway_scale,pathway_projection=projection,pathway_ids=pathways.pathway.tolist(),pathway_provenance=provenance)
    pack['gene_dim']=data.source.shape[1];pack['pathway_dim']=projection.shape[1]
    return pack


class PathwayStructureMLP(nn.Module):
    def __init__(self,gene_dim,pathway_dim,condition_dim,structure_dim,output_dim,use_pathway):
        super().__init__();self.gene_dim=gene_dim;self.use_pathway=use_pathway
        self.expression_encoder=nn.Sequential(nn.Linear(gene_dim,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,64),nn.SiLU())
        self.structure_encoder=nn.Sequential(nn.Linear(structure_dim,64,bias=False),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,32,bias=False),nn.SiLU())
        self.pathway_encoder=nn.Sequential(nn.Linear(pathway_dim,64,bias=False),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,32,bias=False),nn.SiLU())
        self.decoder=nn.Sequential(nn.Linear(64+32+32+condition_dim,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,output_dim))

    def forward(self,source,condition,structure):
        genes=source[:,:self.gene_dim];pathways=source[:,self.gene_dim:]
        if not self.use_pathway:pathways=torch.zeros_like(pathways)
        return self.decoder(torch.cat([self.expression_encoder(genes),self.structure_encoder(structure),self.pathway_encoder(pathways),condition],dim=1))


def run_pathway_comparison(a):
    projection,pathways,provenance=load_projection(a.annotation,a.prepared)
    a.out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(projection_sha256=hashlib.sha256(np.ascontiguousarray(projection).tobytes()).hexdigest(),protocol='matched four-branch MLP ablation; structure/pathway slots zeroed when absent',pathway_score='gene-weighted mean of training-standardized in vitro expression, then training-fitted pathway standardization',architecture='expression128_64_structure64_32_pathway64_32_fusion64_fullprobe')
    config_path=a.out/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=config:raise ValueError('Resume configuration mismatch')
    else:
        if any(a.out.iterdir()):raise ValueError('Unrecognized output directory')
        config_path.write_text(json.dumps(config,indent=2))
    progress_path=a.out/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.is_file() else {'groups':{}}
    def save_progress():
        temporary=progress_path.with_suffix('.tmp');temporary.write_text(json.dumps(progress,indent=2));temporary.replace(progress_path)
    pathways.to_csv(a.out/'pathway_audit.csv',index=False)
    (a.out/'pathway_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print(f'Pathways={len(pathways)}; probes in retained pathways={int(np.any(projection>0,axis=1).sum())}',flush=True)
    tables=[];split_reference=None
    for enabled in [False,True]:
        key='with_pathway' if enabled else 'without_pathway';entry=progress['groups'].get(key)
        if entry and (Path(entry['path'])/'comparison.csv').is_file():
            folder=Path(entry['path']);result=pd.read_csv(folder/'comparison.csv')
            print(key,': completed group reused',flush=True)
        else:
            if entry:print(key,': partial folder preserved; this two-model group restarts',flush=True)
            folder=a.out/f'{key}_{uuid.uuid4().hex[:6]}'
            progress['groups'][key]={'path':str(folder),'status':'running'};save_progress()
            values={**vars(a),'out':folder,'pathway_enabled':enabled,'architecture':config['architecture']}
            group_args=SimpleNamespace(**values)
            def prepare_fn(data,tr,ev):return prepare_pathways(data,tr,ev,projection,pathways,provenance)
            def factory(data,pack,seed):
                first.seed_all(seed)
                return PathwayStructureMLP(pack['gene_dim'],pack['pathway_dim'],pack['conditions'].shape[1],pack['structure'].shape[1],data.target.shape[1],enabled)
            print(f'\nStarting {key}: structure off/on, output={folder}',flush=True)
            result=first.run(group_args,model_factory=factory,prepare_fn=prepare_fn)
        if set(result.model)!={'expression_only','expression_structure'}:raise ValueError('Incomplete model group')
        split=pd.read_csv(folder/'compound_split.csv').sort_values('compound').reset_index(drop=True)
        if split_reference is None:split_reference=split
        else:pd.testing.assert_frame_equal(split_reference,split)
        group_cfg=json.loads((folder/'config.json').read_text())
        if group_cfg.get('pathway_enabled')!=enabled:raise ValueError('Cached pathway flag mismatch')
        for field in ['seed','epochs','check_every','batch_size']:
            if group_cfg[field]!=getattr(a,field):raise ValueError('Cached group configuration mismatch')
        result['pathway']=enabled;result['structure']=result.model.eq('expression_structure')
        result['model']=['expression+structure+pathway' if enabled and s else 'expression+pathway' if enabled else 'expression+structure' if s else 'expression' for s in result.structure]
        tables.append(result)
        progress['groups'][key]['status']='complete';save_progress()
        pd.concat(tables,ignore_index=True).to_csv(a.out/'comparison.csv',index=False)
    result=pd.concat(tables,ignore_index=True)
    if result.parameters.nunique()!=1:raise ValueError('Unmatched parameter counts')
    keys=['n_target_conditions','n_treated_conditions','zero_response_rmse']
    for key in keys:
        if not np.allclose(result[key],result[key].iloc[0]):raise ValueError('Evaluation targets/baseline mismatch')
    result.to_csv(a.out/'comparison.csv',index=False)
    print(result.to_string(index=False));print('Saved:',a.out,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['prepared','structure','annotation','out']:p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--check-every',type=int,default=5);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if min(a.epochs,a.check_every,a.batch_size,a.threads)<1:p.error('Positive settings required')
    with threadpool_limits(limits=a.threads):run_pathway_comparison(a)
