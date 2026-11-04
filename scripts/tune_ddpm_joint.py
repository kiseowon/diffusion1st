"""Validation-only joint DDPM tuning: 12 seed-42 trials, then repeat finalists.

Never scores outer test, refits final models, or launches training without --execute.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import platform
import uuid

import repeated_latent_validation as core
import validate_ddpm_architecture as width


def candidates():
    values = []
    for hidden, lr, wd in itertools.product([64, 128], [.0001, .0003, .001], [0., .01]):
        reference = hidden == 128 and lr == .0003 and wd == .01
        values.append(dict(name='reference' if reference else f'h{hidden}_lr{lr:g}_wd{wd:g}',
                           context=64, hidden=hidden, learning_rate=lr, weight_decay=wd))
    return sorted(values, key=lambda c: c['name'] != 'reference')


def shortlist(results, top_k):
    # Stable tie order follows the prespecified candidate list (reference first).
    order = {c['name']: i for i, c in enumerate(candidates())}
    ranked = sorted(results, key=lambda r: (r['validation_response_rmse'], order[r['candidate']]))
    selected = [r['candidate'] for r in ranked[:top_k]]
    if 'reference' not in selected:
        selected.append('reference')
    return selected


def execute(args):
    import numpy as np
    import pandas as pd
    import torch
    import sklearn
    import scipy
    import joblib
    from threadpoolctl import threadpool_limits
    choices = candidates()
    inputs = [args.prepared / name for name in ['vitro_expression.npy', 'vivo_expression.npy',
              'vitro_conditions.csv', 'vivo_conditions.csv', 'pairs.csv.gz', 'probes.csv']] + [args.structure]
    core.protect_output(args.out, [args.prepared, args.structure])
    sources = [Path(__file__), Path(core.__file__), Path(width.__file__), core.SNAPSHOT]
    contract = dict(purpose='validation_only_joint_hyperparameter_screening_and_seed_repeats',
        candidates=choices, base_protocol=copy.deepcopy(core.PROTOCOL),
        varied_fields=['hidden', 'learning_rate', 'weight_decay'],
        arguments={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'execute'},
        input_sha256={str(p.resolve()): core.digest(p) for p in inputs},
        source_sha256={str(p.resolve()): core.digest(p) for p in sources},
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      torch=torch.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__, joblib=joblib.__version__))
    out = args.out.resolve()
    core.contract_folder(out, contract)
    first, data, ids, meta, Y, taudit, saudit = core.load_data(args)
    mapping = core.compound_folds(data.names, args.split_seed, args.folds)
    train_names, val_names = core.inner_compounds(mapping, args.fold, args.split_seed)
    training = np.flatnonzero(meta.compound.isin(train_names))
    validation = np.flatnonzero(meta.compound.isin(val_names))
    pd.DataFrame([dict(compound=n, stage='inner_train' if n in train_names else
        'inner_validation' if n in val_names else 'outer_test_not_evaluated') for n in mapping]).to_csv(out / 'compound_split.csv', index=False)
    pd.DataFrame(taudit).to_csv(out / 'response_target_audit.csv', index=False)
    pd.DataFrame(saudit).to_csv(out / 'source_delta_audit.csv', index=False)
    data.structure_audit.to_csv(out / 'structure_audit.csv', index=False)
    torch.set_num_threads(args.threads)
    results = []
    with threadpool_limits(limits=args.threads):
        bhistory = []
        for alpha in core.PROTOCOL['baseline_alphas']:
            b, _ = core.baseline_stage(meta, Y, training, alpha)
            bhistory.append(dict(alpha=alpha, validation_response_rmse=core.rmse(Y[validation], b[validation])))
        alpha = min(bhistory, key=lambda r: r['validation_response_rmse'])['alpha']
        baseline, ridge = core.baseline_stage(meta, Y, training, alpha)
        pd.DataFrame(bhistory).to_csv(out / 'baseline_validation.csv', index=False)
        joblib.dump(ridge, out / 'inner_ridge.joblib')
        pack = core.prepare_stage(first, data, ids, meta, Y, training, baseline)
        joblib.dump(pack['state'], out / 'inner_preprocess.joblib')
        np.savez_compressed(out / 'validation_baseline.npz', target_ids=ids[validation], prediction=baseline[validation])
        def trial(candidate, seed, phase):
            job = dict(candidate=candidate['name'], model_seed=seed, phase=phase)
            task = out / f'{candidate["name"]}_seed{seed}'
            task.mkdir(exist_ok=True)
            complete = core.read_completion(task, job)
            if complete is None:
                attempt = task / ('attempt_' + uuid.uuid4().hex)
                attempt.mkdir()
                result = width.fit_trial(candidate, seed, pack, Y, training, validation, baseline, attempt)
                complete = dict(job=job, result=result,
                    sha256={str(p.relative_to(task)): core.digest(p) for p in attempt.iterdir() if p.is_file()})
                core.save_json(task / 'complete.json', complete)
            result = dict(**complete['result'], phase=phase)
            results.append(result)
            pd.DataFrame(results).to_csv(out / 'trials.csv', index=False)
            print(f'Completed {len(results)} trials: {job}', flush=True)
            return result
        print('STAGE 1: joint candidates on the screening initialization', flush=True)
        screening = [trial(c, args.screen_seed, 'screen') for c in choices]
        pd.DataFrame(screening).sort_values('validation_response_rmse', kind='stable').to_csv(out / 'screening.csv', index=False)
        finalists = shortlist(screening, args.top_k)
        core.save_json(out / 'shortlist.json', dict(candidates=finalists, selected_only_from_inner_validation=True,
            policy=f'top_{args.top_k}_plus_original_reference', outer_test_scored=False))
        print('STAGE 2 finalists:', finalists, flush=True)
        for c in choices:
            if c['name'] in finalists:
                for seed in args.repeat_seeds:
                    trial(c, seed, 'initialization_repeat')
    trials = pd.DataFrame(results)
    reference = trials.loc[trials.candidate.eq('reference')].set_index('model_seed').validation_response_rmse
    trials['paired_delta_vs_reference'] = trials.validation_response_rmse - trials.model_seed.map(reference)
    trials.to_csv(out / 'trials.csv', index=False)
    final = trials.loc[trials.candidate.isin(finalists)]
    expected_seeds = {args.screen_seed, *args.repeat_seeds}
    for name, group in final.groupby('candidate'):
        if set(group.model_seed) != expected_seeds or len(group) != len(expected_seeds):
            raise ValueError('Finalist seed coverage differs')
    summary = final.groupby(['candidate', 'context', 'hidden', 'learning_rate', 'weight_decay'], sort=False).agg(
        mean_validation_rmse=('validation_response_rmse', 'mean'),
        std_validation_rmse=('validation_response_rmse', 'std'),
        mean_paired_delta_vs_reference=('paired_delta_vs_reference', 'mean'),
        model_seeds=('model_seed', 'count'), parameter_count=('parameter_count', 'first')).reset_index()
    summary.to_csv(out / 'finalist_summary.csv', index=False)
    selected = summary.loc[summary.mean_validation_rmse.idxmin()].to_dict()
    core.save_json(out / 'selected_candidate.json', dict(selected=selected,
        selection='lowest_mean_validation_rmse_among_screened_finalists',
        claim='validation_selection_not_global_optimum_or_independent_generalization', outer_test_scored=False))
    core.save_json(out / 'completed.json', dict(completed_trials=len(results), finalists=len(finalists), outer_test_scored=False))
    print('JOINT VALIDATION COMPLETE:', out, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'structure', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--screen-seed', type=int, default=42)
    p.add_argument('--repeat-seeds', type=int, nargs='+', default=[43, 44])
    p.add_argument('--top-k', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    seeds = [args.screen_seed, *args.repeat_seeds]
    if args.folds < 2 or not 0 <= args.fold < args.folds or args.threads < 1 or not 1 <= args.top_k <= len(candidates()):
        p.error('Invalid fold, threads or shortlist size')
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in [args.split_seed, *seeds]):
        p.error('Screen/repeat seeds must be disjoint, unique and in [0, 2**32)')
    max_finalists = min(args.top_k + 1, len(candidates()))
    print(json.dumps(dict(mode='execute' if args.execute else 'plan_only', candidates=candidates(),
        screen_seed=args.screen_seed, repeat_seeds=args.repeat_seeds,
        screen_fits=len(candidates()), total_inner_fits_up_to=len(candidates()) + max_finalists * len(args.repeat_seeds),
        outer_test_evaluations=0, final_refits=0), indent=2), flush=True)
    if args.execute:
        execute(args)


if __name__ == '__main__':
    main()
