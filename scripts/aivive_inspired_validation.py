"""AIVIVE-inspired, PCA-free response prediction; not an AIVIVE reproduction."""
import argparse
from pathlib import Path
import json
import repeated_latent_validation as core

SETTINGS = dict(protocol_name='aivive_inspired_response_v1', input_components=None,
                output_components=None, input_transform='training_source_probe_MinMax',
                output_transform='training_target_probe_MinMax_no_clipping',
                context=256, hidden=64, activation='LeakyReLU_0.2', dropout=0.0,
                learning_rate=1e-5, weight_decay=0., optimizer='Adam',
                discriminator_optimizer='SGD_lr_0.0001_momentum_0.9',
                batch_size=64, epochs=100, validation_epochs=[1,25,50,75,100],
                mlp_loss='scaled_response_MAE', cgan_loss='10_scaled_MAE_plus_2_BCE_no_cycle',
                pathway=False, pca=False, output_activation='linear')


def prepare(first, data, ids, meta, Y, fit, baseline, use_structure=True, prediction_target='direct_response'):
    import numpy as np
    import torch
    from sklearn.preprocessing import MinMaxScaler
    if prediction_target != 'direct_response' or np.any(baseline != 0):
        raise ValueError('Only direct response with no Ridge offset is supported')
    rows = np.flatnonzero(np.isin(data.target_ids, ids[fit]))
    # Reuse the reviewed condition encoding; replace source StandardScaler entirely.
    pack = first.prepare(data, rows, rows)
    source_scaler = MinMaxScaler().fit(data.source[np.unique(data.source_ids[rows])])
    source = source_scaler.transform(data.source)
    E = np.asarray([source[data.source_ids[data.target_ids == cid]].mean(0) for cid in ids])
    C = np.asarray([pack['conditions'][data.target_ids == cid].numpy().mean(0) for cid in ids])
    S = data.structure[[data.compound_to_index[c] for c in meta.compound]]
    if not use_structure:
        S = np.zeros_like(S)
    X = np.c_[E, C, S].astype(np.float32)
    output_scaler = MinMaxScaler().fit(Y[fit])
    Z = torch.zeros(Y.shape, dtype=torch.float32)
    Z[fit] = torch.tensor(output_scaler.transform(Y[fit]), dtype=torch.float32)
    if not np.isfinite(X).all() or not torch.isfinite(Z).all():
        raise ValueError('Nonfinite input or output')
    return dict(X=torch.tensor(X), Z=Z, state=dict(source_scaler=source_scaler,
                condition_preprocess=pack['preprocess']['condition_encoder'], output_scaler=output_scaler,
                prediction_target='direct_response', structure_enabled=use_structure, pca=False))


def restore(scores, state):
    return state['output_scaler'].inverse_transform(scores)


def models(kind, input_dim, output_dim, seed):
    import random
    import numpy as np
    import torch
    from torch import nn
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = output_dim
            self.context = nn.Sequential(nn.Linear(input_dim, SETTINGS['context']), nn.LeakyReLU(.2))
            self.time = nn.Embedding(core.PROTOCOL['steps'],16)
            extra = core.PROTOCOL['cgan_noise_dim'] if kind == 'cgan' else output_dim + 16
            self.head = nn.Sequential(nn.Linear(SETTINGS['context']+extra,SETTINGS['hidden']),
                                      nn.LeakyReLU(.2),nn.Linear(SETTINGS['hidden'],output_dim))
        def forward(self,x,noisy,t=None):
            import torch
            extra = noisy if kind == 'cgan' else torch.cat([noisy,self.time(t)],dim=1)
            return self.head(torch.cat([self.context(x),extra],dim=1))
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.context = nn.Sequential(nn.Linear(input_dim,256),nn.ReLU())
            self.head = nn.Sequential(nn.Linear(256+output_dim,64),nn.ReLU(),nn.Linear(64,1))
        def forward(self,x,z):
            return self.head(torch.cat([self.context(x),z],dim=1)).squeeze(1)
    model=Predictor()
    opt=torch.optim.Adam(model.parameters(),lr=core.PROTOCOL['learning_rate'])
    if kind != 'cgan':
        return model,opt,None,None
    disc=Discriminator()
    return model,opt,disc,torch.optim.SGD(disc.parameters(),lr=1e-4,momentum=.9)


def train(kind, model, opt, disc, dopt, pack, fit):
    import torch
    import torch.nn.functional as F
    if kind == 'ddpm':
        return core.train_epoch(kind, model,opt,pack,fit)
    model.train()
    if disc is not None:
        disc.train()
    for local in torch.randperm(len(fit)).split(core.PROTOCOL['batch_size']):
        rows=fit[local.numpy()]; x=pack['X'][rows]; truth=pack['Z'][rows]
        if kind == 'mlp':
            pred=model(x,torch.zeros_like(truth),torch.zeros(len(rows),dtype=torch.long))
            loss=F.l1_loss(pred,truth)
        else:
            for p in disc.parameters(): p.requires_grad_(True)
            fake=model(x,torch.randn(len(rows),core.PROTOCOL['cgan_noise_dim'])).detach()
            real_logits=disc(x,truth); fake_logits=disc(x,fake)
            dl=.5*(F.binary_cross_entropy_with_logits(real_logits,torch.ones_like(real_logits))+
                   F.binary_cross_entropy_with_logits(fake_logits,torch.zeros_like(fake_logits)))
            if not torch.isfinite(dl): raise ValueError('Nonfinite discriminator loss')
            dopt.zero_grad(); dl.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(),core.PROTOCOL['gradient_clip']);dopt.step()
            for p in disc.parameters(): p.requires_grad_(False)
            pred=model(x,torch.randn(len(rows),core.PROTOCOL['cgan_noise_dim']))
            logits=disc(x,pred)
            loss=10*F.l1_loss(pred,truth)+2*F.binary_cross_entropy_with_logits(logits,torch.ones_like(logits))
        if not torch.isfinite(loss): raise ValueError('Nonfinite prediction loss')
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),core.PROTOCOL['gradient_clip']);opt.step()


def main(argv=None):
    parser=core.parser()
    parser.description=__doc__
    parser.set_defaults(prediction_target='direct_response',families=['mlp','cgan','ddpm'])
    args=parser.parse_args(argv)
    if args.prediction_target != 'direct_response' or args.no_structure or args.paired_on_run:
        parser.error('This runner executes both structure arms, direct response only')
    if args.folds<2 or args.threads<1 or args.bootstrap_samples<0:
        parser.error('Invalid folds/threads/bootstrap')
    for values in [args.families,args.model_seeds,args.split_seeds]:
        if len(values)!=len(set(values)): parser.error('Duplicate values')
    if any(s<0 or s>=2**32 for s in args.split_seeds+args.model_seeds):
        parser.error('Invalid seed')
    n=2*len(core.jobs(args))
    print(json.dumps(dict(mode='execute' if args.execute else 'plan_only',neural_jobs=n,
                         neural_fits_up_to=2*n,settings=SETTINGS,
                         note='AIVIVE-inspired adaptation, not reproduction or optimal hyperparameters'),indent=2),flush=True)
    if not args.execute: return n
    old=core.PROTOCOL.copy()
    hooks={k:getattr(core,k) for k in ['prepare_stage','restore','make_family_model','train_family_epoch']}
    old_sources=getattr(core,'EXTRA_SOURCE_FILES',[])
    root=args.out
    try:
        core.PROTOCOL.update(SETTINGS)
        core.EXTRA_SOURCE_FILES=[Path(__file__)]
        core.prepare_stage=prepare;core.restore=restore;core.make_family_model=models;core.train_family_epoch=train
        for arm in ['basic','structure']:
            args.out=root/arm;args.no_structure=(arm=='basic')
            core.execute(args)
        import pandas as pd
        summary=pd.concat([pd.read_csv(root/arm/'summary.csv').assign(arm=arm) for arm in ['basic','structure']],ignore_index=True)
        summary.to_csv(root/'structure_summary.csv',index=False)
        print(summary.to_string(index=False),flush=True)
    finally:
        core.PROTOCOL.clear();core.PROTOCOL.update(old)
        for key,value in hooks.items():setattr(core,key,value)
        core.EXTRA_SOURCE_FILES=old_sources
    return n


if __name__=='__main__': main()
