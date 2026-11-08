"""Inspect saved input transforms without fitting models or regenerating predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import repeated_latent_validation as core


def condition_report(table, encoder, fit):
    features=pd.DataFrame(dict(dose=np.log1p(table.dose.to_numpy(float)),time=table.time_h.to_numpy(float),unit=table.unit.astype(str),level=table.level.astype(str)))
    numeric=encoder.named_transformers_['numeric'].transform(features[['dose','time']])
    low=numeric[fit].min(0);high=numeric[fit].max(0)
    units,levels=encoder.named_transformers_['category'].categories_
    return pd.DataFrame(dict(compound=table.compound.to_numpy(),condition_id=table.condition_id.to_numpy(),unit=features.unit.to_numpy(),dose=table.dose.to_numpy(float),time_h=table.time_h.to_numpy(float),scaled_log_dose_abs=np.abs(numeric[:,0]),scaled_time_abs=np.abs(numeric[:,1]),numeric_outside_training=((numeric<low)|(numeric>high)).any(1),unknown_unit=~features.unit.isin(units).to_numpy(),unknown_level=~features.level.isin(levels).to_numpy()))


def inspect(unit, out):
    import sklearn
    info=json.loads((unit/'run.json').read_text())
    prepared=Path(info['arguments']['prepared'])
    core.protect_output(out,[unit,prepared])
    if out.exists():raise ValueError('Use a new diagnostic output folder')
    if info['versions']['sklearn']!=sklearn.__version__:raise ValueError('Use the original sklearn version to load saved preprocessing')
    files=[prepared/n for n in ['vitro_expression.npy','vitro_conditions.csv','vivo_conditions.csv']]
    for path in files:
        expected=info['input_sha256'].get(str(path.resolve()))
        if expected is None or core.digest(path)!=expected:raise ValueError('Saved input checksum mismatch: '+str(path))
    source_table=pd.read_csv(files[1]);target_table=pd.read_csv(files[2])
    raw=np.load(files[0],mmap_mode='r')
    if len(source_table)!=len(raw) or not np.array_equal(source_table.condition_id,np.arange(len(raw))):raise ValueError('Source alignment mismatch')
    if not np.array_equal(target_table.condition_id,np.arange(len(target_table))):raise ValueError('Target alignment mismatch')
    sa=pd.read_csv(unit/'source_control_audit.csv');ta=pd.read_csv(unit/'target_control_audit.csv')
    source=source_table.iloc[sa.condition_id.to_numpy(int)].reset_index(drop=True)
    target=target_table.iloc[ta.condition_id.to_numpy(int)].reset_index(drop=True)
    delta=np.asarray(raw[sa.condition_id.to_numpy(int)]-raw[sa.control_condition_id.to_numpy(int)],dtype=np.float32)
    split=pd.read_csv(unit/'compound_split.csv')
    test=set(split.loc[split.stage.eq('test'),'compound'])
    if not test:raise ValueError('No test compounds')
    reports=[];targets=[];references=[];consumed=files+[unit/n for n in ['run.json','compound_split.csv','source_control_audit.csv','target_control_audit.csv']]
    for stage in ['inner','final']:
        path=unit/'basic'/f'{stage}_preprocess.joblib';consumed.append(path)
        state=joblib.load(path)
        if state.get('input_space')!='pca':raise ValueError('This diagnostic requires saved PCA inputs')
        names=set(split.loc[split.stage.isin(['train'] if stage=='inner' else ['train','validation']),'compound'])
        fit=source.compound.isin(names).to_numpy();tfit=target.compound.isin(names).to_numpy()
        if not fit.any() or not tfit.any():raise ValueError('No fitting rows')
        scaled=state['source_scaler'].transform(delta)
        scores=state['source_score_scaler'].transform(state['source_pca'].transform(scaled)[:,state['source_keep']])
        if not np.isfinite(scores).all():raise ValueError('Nonfinite transformed inputs')
        low=scores[fit].min(0);high=scores[fit].max(0)
        report=condition_report(source,state['source_condition_encoder'],fit)
        report['pca_abs_max']=np.abs(scores).max(1)
        report['pca_outside_training_fraction']=((scores<low)|(scores>high)).mean(1)
        report['probe_outside_MinMax_fraction']=((scaled<0)|(scaled>1)).mean(1)
        slots=pd.DataFrame(state['source_slot_audit']).set_index('condition_id')['slot']
        assigned=source.condition_id.map(slots)
        if assigned.isna().any():raise ValueError('Missing saved slot')
        observed=set(assigned[fit])
        report['slot_not_observed_in_training']=~assigned.isin(observed).to_numpy()
        report['stage']=stage;report['partition']=np.where(fit,'fit',np.where(source.compound.isin(test),'test','validation'))
        reports.append(report)
        tr=condition_report(target,state['target_condition_encoder'],tfit)
        tr['stage']=stage;tr['partition']=np.where(tfit,'fit',np.where(target.compound.isin(test),'test','validation'));targets.append(tr)
        references.append(dict(stage=stage,training_compounds=len(names),training_source_PCA_abs_max=float(np.abs(scores[fit]).max()),training_observed_slots=len(observed),defined_slots=len(state['slots']),output_scale=state['output_scale']))
    source_detail=pd.concat(reports,ignore_index=True);target_detail=pd.concat(targets,ignore_index=True)
    summary=source_detail.groupby(['stage','partition','compound'],as_index=False).agg(units=('unit',lambda x:' | '.join(sorted(set(x)))),pca_abs_max=('pca_abs_max','max'),pca_outside_fraction_mean=('pca_outside_training_fraction','mean'),probe_outside_MinMax_fraction_mean=('probe_outside_MinMax_fraction','mean'),scaled_log_dose_abs_max=('scaled_log_dose_abs','max'),scaled_time_abs_max=('scaled_time_abs','max'),unknown_unit_conditions=('unknown_unit','sum'),unknown_level_conditions=('unknown_level','sum'),unseen_slot_conditions=('slot_not_observed_in_training','sum'))
    out.mkdir(parents=True,exist_ok=False)
    source_detail.to_csv(out/'source_conditions.csv',index=False);target_detail.to_csv(out/'target_conditions.csv',index=False)
    summary.to_csv(out/'source_by_compound.csv',index=False);pd.DataFrame(references).to_csv(out/'training_reference.csv',index=False)
    core.save_json(out/'provenance.json',dict(unit=str(unit.resolve()),training=False,prediction_regeneration=False,fitted_new_preprocessing=False,limitations='Range exceedance is descriptive, not proof of causality or an error. Numeric dose comparisons are not physical unit conversion. Source/target metadata only; no structure or model dynamics diagnosis.',consumed_sha256={str(p.resolve()):core.digest(p) for p in consumed},diagnostic_source_sha256=core.digest(Path(__file__))))
    print(pd.DataFrame(references).to_string(index=False))
    print('\nFINAL TEST SOURCE INPUTS')
    print(summary[summary.stage.eq('final')&summary.partition.eq('test')].sort_values('pca_abs_max',ascending=False).to_string(index=False))
    print('\nFINAL TEST TARGET CONDITIONS')
    print(target_detail[target_detail.stage.eq('final')&target_detail.partition.eq('test')].groupby('compound')[['scaled_log_dose_abs','scaled_time_abs','unknown_unit','unknown_level']].max().sort_values('scaled_log_dose_abs',ascending=False).to_string())
    print('DIAGNOSTIC COMPLETE:',out)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();inspect(a.unit,a.out)

if __name__=='__main__':main()
