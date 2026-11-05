"""MLP basic/2D-descriptor/Morgan comparison, using saved SMILES only; no PubChem calls."""
from pathlib import Path
import json
import repeated_latent_validation as core
import aivive_inspired_validation as inspired


def calculate_descriptors(smiles_csv):
    import numpy as np
    import pandas as pd
    from rdkit import Chem, rdBase
    from rdkit.Chem import Descriptors, Descriptors3D
    table=pd.read_csv(smiles_csv)
    if not {'compound','smiles'}.issubset(table.columns):
        raise ValueError('SMILES CSV requires compound and smiles columns')
    if table.compound.isna().any(): raise ValueError('Missing compound name')
    table.compound=table.compound.astype(str).str.strip()
    if table.compound.duplicated().any(): raise ValueError('Duplicate SMILES compound names')
    # RDKit Descriptors registry is 2D: no conformer generation or Descriptors3D registry.
    three_d={n for n,_ in Descriptors3D.descList}
    functions=sorted([(n,f) for n,f in Descriptors.descList if n not in three_d],key=lambda x:x[0])
    names=[n for n,_ in functions]
    if len(set(names))!=len(names): raise ValueError('Duplicate descriptor names')
    rows=[]; audit=[]
    for row in table.itertuples():
        mol=Chem.MolFromSmiles(row.smiles) if isinstance(row.smiles,str) else None
        values=[]
        for name,fn in functions:
            try: value=float(fn(mol)) if mol is not None else np.nan
            except Exception: value=np.nan
            values.append(value if np.isfinite(value) else np.nan)
        rows.append(values)
        audit.append(dict(compound=row.compound,parsed=mol is not None,
                          nonfinite_descriptors=int(np.isnan(values).sum())))
    return pd.DataFrame(rows,index=table.compound,columns=names),pd.DataFrame(audit),rdBase.rdkitVersion


def fit_descriptors(raw, fit_names, threshold=.95):
    """Select/scale unique training compounds only; deterministic greedy correlation filter."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    present=raw.notna().any(axis=1)
    names=sorted(set(map(str,fit_names)) & set(raw.index[present]))
    if len(names)<3: raise ValueError('Need at least 3 training compounds with descriptors')
    training=raw.loc[names]
    finite=np.isfinite(training.to_numpy()).all(axis=0)
    varying=np.ptp(training.to_numpy(),axis=0)>0
    candidates=list(training.columns[finite & varying])
    if not candidates: raise ValueError('No finite nonconstant training descriptors')
    corr=training[candidates].corr().abs()
    kept=[]; dropped={}
    for name in candidates:
        partner=next((old for old in kept if corr.loc[old,name]>threshold),None)
        if partner is None: kept.append(name)
        else: dropped[name]=partner
    scaler=StandardScaler().fit(training[kept].to_numpy(float))
    return dict(selected=kept,scaler=scaler,training_compounds=names,
                dropped_nonfinite=list(training.columns[~finite]),
                dropped_constant=list(training.columns[finite & ~varying]),
                dropped_correlated=dropped,threshold=threshold)


def encode_descriptors(raw,names,state,width):
    import numpy as np
    values=raw.reindex(names)[state['selected']].to_numpy(float)
    present=raw.reindex(names).notna().any(axis=1).to_numpy()
    bad=~np.isfinite(values)
    # Held-out missing selected values use training mean (scaled zero), never test mean.
    values=np.where(bad,state['scaler'].mean_[None,:],values)
    output=np.zeros((len(names),width),dtype=np.float32)
    output[:,:len(state['selected'])]=state['scaler'].transform(values)
    output[~present,:]=0
    output[:,-1]=present.astype(np.float32)
    return output,bad.sum(axis=1)


def main(argv=None):
    p=core.parser();p.description=__doc__
    p.add_argument('--smiles',type=Path,required=True)
    p.set_defaults(families=['mlp'],prediction_target='direct_response')
    a=p.parse_args(argv)
    if a.families!=['mlp'] or a.prediction_target!='direct_response' or a.no_structure or a.paired_on_run:
        p.error('MLP direct-response three-arm runner only')
    if a.folds<2 or a.threads<1 or a.bootstrap_samples<0: p.error('Invalid count')
    for values in [a.split_seeds,a.model_seeds]:
        if len(values)!=len(set(values)) or any(s<0 or s>=2**32 for s in values):p.error('Invalid seeds')
    count=3*len(core.jobs(a))
    print(json.dumps(dict(mode='execute' if a.execute else 'plan_only',arms=['basic','descriptor','morgan'],
                         neural_jobs=count,neural_fits_up_to=2*count,
                         protocol='same_AIVIVE_inspired_PCA_free_MLP',correlation_threshold=.95),indent=2),flush=True)
    if not a.execute: return count
    raw,audit,version=calculate_descriptors(a.smiles)
    import numpy as np
    import pandas as pd
    import torch
    root=a.out
    core.protect_output(root,[a.prepared,a.structure,a.smiles])
    old=core.PROTOCOL.copy()
    hooks={k:getattr(core,k) for k in ['prepare_stage','restore','make_family_model','train_family_epoch']}
    old_sources=getattr(core,'EXTRA_SOURCE_FILES',[]);old_inputs=getattr(core,'EXTRA_INPUT_FILES',[])
    try:
        core.PROTOCOL.update(inspired.SETTINGS)
        core.PROTOCOL.update(protocol_name='AIVIVE_inspired_representation_comparison_v1',
            descriptor_names=list(raw.columns),rdkit_version=version,descriptor_correlation=.95,
            descriptor_molecule_policy='full_saved_SMILES_no_fragment_selection_no_3D',
            descriptor_missing_policy='drop_train_nonfinite_columns_test_training_mean',
            structure_slots='shared_padded_same_parameter_count')
        core.EXTRA_SOURCE_FILES=[Path(__file__),Path(inspired.__file__)]
        core.EXTRA_INPUT_FILES=[a.smiles]
        core.restore=inspired.restore;core.make_family_model=inspired.models;core.train_family_epoch=inspired.train
        for arm in ['basic','descriptor','morgan']:
            core.PROTOCOL['structure_representation']=arm
            def prepare(*pos, _arm=arm, **kwargs):
                kwargs['use_structure']=True
                pack=inspired.prepare(*pos,**kwargs)
                data,ids,meta,fit=pos[1],pos[2],pos[3],pos[5]
                original_width=data.structure.shape[1]
                width=max(original_width,len(raw.columns)+1)
                state=fit_descriptors(raw,meta.iloc[fit].compound)
                names=meta.compound.astype(str).tolist()
                structural=np.zeros((len(names),width),dtype=np.float32)
                missing=np.zeros(len(names),dtype=int)
                if _arm=='descriptor':
                    structural,missing=encode_descriptors(raw,names,state,width)
                elif _arm=='morgan':
                    original=data.structure[[data.compound_to_index[n] for n in names]]
                    structural[:,:original_width-1]=original[:,:-1]
                    structural[:,-1]=original[:,-1]
                pack['X']=torch.cat([pack['X'][:,:-original_width],torch.tensor(structural)],dim=1)
                pack['state'].update(structure_representation=_arm,descriptor_preprocess=state,
                                     structure_slots=width,descriptor_missing_values_by_target=missing)
                return pack
            core.prepare_stage=prepare
            a.out=root/arm;a.no_structure=(arm=='basic')
            core.execute(a)
        raw.to_csv(root/'raw_2d_descriptors.csv',index_label='compound')
        audit.to_csv(root/'descriptor_calculation_audit.csv',index=False)
        coverage=pd.read_csv(root/'morgan'/'structure_audit.csv').rename(columns={'structure_present':'morgan_present'})
        coverage['descriptor_present']=coverage.compound.map(raw.notna().any(axis=1)).fillna(False).astype(int)
        coverage.to_csv(root/'structure_coverage.csv',index=False)
        summaries=[];details=[]
        for arm in ['basic','descriptor','morgan']:
            summaries.append(pd.read_csv(root/arm/'summary.csv').assign(arm=arm))
            details.append(pd.read_csv(root/arm/'oof_condition_metrics.csv').assign(arm=arm))
        summary=pd.concat(summaries,ignore_index=True);summary.to_csv(root/'representation_summary.csv',index=False)
        detail=pd.concat(details,ignore_index=True)
        units=detail.groupby(['arm','split_seed','model_seed','model']).response_mse.mean().pow(.5).reset_index(name='response_rmse')
        units.to_csv(root/'representation_by_seed.csv',index=False)
        paired=units[units.model.eq('mlp')].pivot(index=['split_seed','model_seed'],columns='arm',values='response_rmse')
        for left,right in [('descriptor','basic'),('morgan','basic'),('descriptor','morgan')]:
            paired[left+'_minus_'+right]=paired[left]-paired[right]
        paired.to_csv(root/'representation_paired_differences.csv')
        print(summary.to_string(index=False),flush=True)
    finally:
        core.PROTOCOL.clear();core.PROTOCOL.update(old)
        for k,v in hooks.items():setattr(core,k,v)
        core.EXTRA_SOURCE_FILES=old_sources;core.EXTRA_INPUT_FILES=old_inputs
    return count


if __name__=='__main__':main()
