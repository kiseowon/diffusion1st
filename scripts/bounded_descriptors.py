"""Training-only bounded descriptor preprocessing; fixed numeric policy, not tuned to outcomes."""
import numpy as np


def signed_log(values):
    return np.sign(values)*np.log1p(np.abs(values))


def fit(raw, names, threshold=.95):
    from sklearn.preprocessing import MinMaxScaler
    names=sorted(set(map(str,names)) & set(raw.index[raw.notna().any(axis=1)]))
    if len(names)<3:raise ValueError('Need at least 3 training compounds with descriptors')
    train=raw.loc[names]
    finite=np.isfinite(train.to_numpy(float)).all(0)
    transformed=signed_log(train.to_numpy(float))
    safe=np.where(np.isfinite(transformed),transformed,0.)
    span=np.ptp(safe,axis=0)
    tolerance=32*np.finfo(float).eps*np.maximum(1.,np.max(np.abs(safe),axis=0))
    varying=span>tolerance
    columns=list(train.columns[finite & varying])
    if not columns:raise ValueError('No finite nondegenerate training descriptors')
    import pandas as pd
    values=pd.DataFrame(signed_log(train[columns].to_numpy(float)),columns=columns)
    correlation=values.corr().abs();selected=[];dropped={}
    for col in columns:
        partner=next((old for old in selected if correlation.loc[old,col]>threshold),None)
        if partner is None:selected.append(col)
        else:dropped[col]=partner
    selected_values=signed_log(train[selected].to_numpy(float))
    scaler=MinMaxScaler(feature_range=(-1,1),clip=True).fit(selected_values)
    # Held-out nonfinite selected entries use an in-range training median.
    medians=np.median(selected_values,axis=0)
    return dict(policy='signed_log_train_MinMax_clip_minus1_plus1_v1',selected=selected,
        scaler=scaler,training_medians=medians,training_compounds=names,
        dropped_nonfinite=list(train.columns[~finite]),
        dropped_constant_or_numerically_degenerate=list(train.columns[finite & ~varying]),
        dropped_correlated=dropped,correlation_threshold=threshold)


def encode(raw,names,state,width):
    values=raw.reindex(names)[state['selected']].to_numpy(float)
    bad=~np.isfinite(values);transformed=signed_log(values)
    transformed=np.where(bad,state['training_medians'][None,:],transformed)
    present=raw.reindex(names).notna().any(axis=1).to_numpy()
    scaler=state['scaler']
    outside=(transformed<scaler.data_min_) | (transformed>scaler.data_max_)
    scaled=scaler.transform(transformed)
    output=np.zeros((len(names),width),dtype=np.float32)
    output[:,:len(state['selected'])]=scaled
    output[~present]=0;output[:,-1]=present
    if not np.isfinite(output).all() or np.max(np.abs(output))>1.00001:
        raise ValueError('Descriptor numeric guard failed')
    return output,dict(missing_selected_values=bad.sum(1),
        clipped_selected_values=np.where(present,outside.sum(1),0),
        selected_descriptor_count=len(state['selected']))
