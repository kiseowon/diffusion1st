"""Plan an internal-validation-only DDPM width sensitivity study. Never scores outer test.

Fixed input/output PCA and training budget; five one-factor-at-a-time candidates.
Use --execute for training. Default is plan-only without reading inputs.
"""
import argparse
import copy
import json
from pathlib import Path
import platform
import uuid

import repeated_latent_validation as core

CANDIDATES = [dict(name='reference', context=64, hidden=128),
              dict(name='context32', context=32, hidden=128),
              dict(name='context128', context=128, hidden=128),
              dict(name='hidden64', context=64, hidden=64),
              dict(name='hidden256', context=64, hidden=256)]


def model_for(candidate, pack, seed):
    import torch
    from torch import nn
    import numpy as np
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    input_dim, latent_dim = pack['X'].shape[1], pack['Z'].shape[1]
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim = latent_dim
            self.context = nn.Sequential(nn.Linear(input_dim, candidate['context']), nn.SiLU())
            self.time = nn.Embedding(core.PROTOCOL['steps'], 16)
            self.head = nn.Sequential(nn.Linear(candidate['context'] + latent_dim + 16, candidate['hidden']),
                                      nn.SiLU(), nn.Linear(candidate['hidden'], latent_dim))
        def forward(self, x, noisy, t):
            return self.head(torch.cat([self.context(x), noisy, self.time(t)], dim=1))
    model = Predictor()
    opt = torch.optim.AdamW(model.parameters(), lr=core.PROTOCOL['learning_rate'],
                           weight_decay=core.PROTOCOL['weight_decay'])
    return model, opt


def fit_trial(candidate, seed, pack, Y, training, validation, baseline, folder):
    import numpy as np
    import pandas as pd
    import torch
    model, opt = model_for(candidate, pack, seed)
    best = dict(epoch=0, strength=0., validation_response_rmse=core.rmse(Y[validation], baseline[validation]))
    history = [best.copy()]
    for epoch in range(1, core.PROTOCOL['epochs'] + 1):
        core.train_epoch('ddpm', model, opt, pack, training)
        if epoch in core.PROTOCOL['validation_epochs']:
            latent = core.predict_scores('ddpm', model, pack, validation, seed + 20000)
            correction = core.restore(latent, pack['state'])
            for strength in core.PROTOCOL['strengths']:
                row = dict(epoch=epoch, strength=strength,
                           validation_response_rmse=core.rmse(Y[validation], baseline[validation] + strength * correction))
                if not np.isfinite(row['validation_response_rmse']):
                    raise ValueError('Nonfinite validation score')
                history.append(row)
                if row['validation_response_rmse'] < best['validation_response_rmse']:
                    best = row.copy()
                    torch.save(model.state_dict(), folder / 'best_validation_model.pt')
            pd.DataFrame(history).to_csv(folder / 'validation_history.csv', index=False)
            print(f'{candidate["name"]} seed={seed} epoch={epoch}: {best}', flush=True)
    result = dict(candidate=candidate['name'], context=candidate['context'], hidden=candidate['hidden'],
                  model_seed=seed, parameter_count=sum(p.numel() for p in model.parameters()), **best)
    core.save_json(folder / 'selection.json', result)
    return result


def execute(args):
    import numpy as np
    import pandas as pd
    import torch
    import sklearn
    import scipy
    import joblib
    from threadpoolctl import threadpool_limits
    inputs = [args.prepared / name for name in ['vitro_expression.npy', 'vivo_expression.npy',
              'vitro_conditions.csv', 'vivo_conditions.csv', 'pairs.csv.gz', 'probes.csv']] + [args.structure]
    core.protect_output(args.out, [args.prepared, args.structure])
    contract = dict(purpose='internal_validation_sensitivity_not_outer_test_evaluation',
        candidates=CANDIDATES, fixed_protocol=copy.deepcopy(core.PROTOCOL),
        arguments={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'execute'},
        input_sha256={str(p.resolve()): core.digest(p) for p in inputs},
        source_sha256={str(p.resolve()): core.digest(p) for p in [Path(__file__), Path(core.__file__), core.SNAPSHOT]},
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      torch=torch.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, joblib=joblib.__version__))
    out = args.out.resolve()
    core.contract_folder(out, contract)
    first, data, ids, meta, Y, target_audit, source_audit = core.load_data(args)
    mapping = core.compound_folds(data.names, args.split_seed, args.folds)
    train_names, val_names = core.inner_compounds(mapping, args.fold, args.split_seed)
    training = np.flatnonzero(meta.compound.isin(train_names))
    validation = np.flatnonzero(meta.compound.isin(val_names))
    # Outer-test labels are not used for fitting, scoring, candidate selection, or reporting.
    pd.DataFrame([dict(compound=n, stage='inner_train' if n in train_names else
        'inner_validation' if n in val_names else 'outer_test_not_evaluated') for n in mapping]).to_csv(out / 'compound_split.csv', index=False)
    pd.DataFrame(target_audit).to_csv(out / 'response_target_audit.csv', index=False)
    pd.DataFrame(source_audit).to_csv(out / 'source_delta_audit.csv', index=False)
    data.structure_audit.to_csv(out / 'structure_audit.csv', index=False)
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        baseline_history = []
        for alpha in core.PROTOCOL['baseline_alphas']:
            baseline, _ = core.baseline_stage(meta, Y, training, alpha)
            baseline_history.append(dict(alpha=alpha, validation_response_rmse=core.rmse(Y[validation], baseline[validation])))
        alpha = min(baseline_history, key=lambda r: r['validation_response_rmse'])['alpha']
        baseline, ridge = core.baseline_stage(meta, Y, training, alpha)
        pd.DataFrame(baseline_history).to_csv(out / 'baseline_validation.csv', index=False)
        joblib.dump(ridge, out / 'inner_ridge.joblib')
        pack = core.prepare_stage(first, data, ids, meta, Y, training, baseline)
        joblib.dump(pack['state'], out / 'inner_preprocess.joblib')
        np.savez_compressed(out / 'validation_baseline.npz', target_ids=ids[validation], prediction=baseline[validation])
        results = []
        for candidate in CANDIDATES:
            for seed in args.model_seeds:
                job = dict(candidate=candidate['name'], model_seed=seed)
                task = out / f'{candidate["name"]}_seed{seed}'
                task.mkdir(exist_ok=True)
                complete = core.read_completion(task, job)
                if complete is None:
                    attempt = task / ('attempt_' + uuid.uuid4().hex)
                    attempt.mkdir()
                    result = fit_trial(candidate, seed, pack, Y, training, validation, baseline, attempt)
                    complete = dict(job=job, result=result, sha256={str(p.relative_to(task)): core.digest(p) for p in attempt.iterdir() if p.is_file()})
                    core.save_json(task / 'complete.json', complete)
                results.append(complete['result'])
                pd.DataFrame(results).to_csv(out / 'trials.csv', index=False)
                print(f'Completed {len(results)}/{len(CANDIDATES) * len(args.model_seeds)}', flush=True)
    trials = pd.DataFrame(results)
    summary = trials.groupby(['candidate', 'context', 'hidden'], sort=False).agg(
        mean_validation_rmse=('validation_response_rmse', 'mean'),
        std_validation_rmse=('validation_response_rmse', 'std'),
        model_seeds=('model_seed', 'count'), parameter_count=('parameter_count', 'first')).reset_index()
    reference = trials.loc[trials.candidate.eq('reference')].set_index('model_seed').validation_response_rmse
    trials['paired_delta_vs_reference'] = trials.validation_response_rmse - trials.model_seed.map(reference)
    trials.to_csv(out / 'trials.csv', index=False)
    summary.to_csv(out / 'candidate_summary.csv', index=False)
    selected = summary.loc[summary.mean_validation_rmse.idxmin()].to_dict()
    core.save_json(out / 'selected_candidate.json', dict(selected=selected,
        selection='lowest_mean_inner_validation_rmse_across_model_seeds',
        claim='best_within_five_candidates_on_this_validation_only_not_global_optimal',
        outer_test_scored=False))
    core.save_json(out / 'completed.json', dict(completed_trials=len(trials), outer_test_scored=False))
    print('VALIDATION STUDY COMPLETE:', out, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'structure', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--model-seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    if args.folds < 2 or not 0 <= args.fold < args.folds or args.threads < 1:
        p.error('Invalid fold or thread count')
    if len(set(args.model_seeds)) != len(args.model_seeds) or any(s < 0 or s >= 2**32 for s in [args.split_seed, *args.model_seeds]):
        p.error('Seeds must be unique and in [0, 2**32)')
    print(json.dumps(dict(mode='execute' if args.execute else 'plan_only', candidates=CANDIDATES,
        inner_validation_fold=args.fold, split_seed=args.split_seed, model_seeds=args.model_seeds,
        inner_fits=len(CANDIDATES) * len(args.model_seeds), refits=0, outer_test_evaluations=0), indent=2), flush=True)
    if args.execute:
        execute(args)


if __name__ == '__main__':
    main()
