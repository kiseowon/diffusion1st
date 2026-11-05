"""Compare completed structure-on/off runs after checking paired experimental contracts.

Read-only with respect to training runs. No training, inference or model deserialization.
"""
import argparse
import json
from pathlib import Path

import repeated_latent_validation as core

# Reviewed structure-on implementation used by the existing three-family run.
LEGACY_ON_RUNNER_SHA256 = '7f945575b8139d380b67bad4d38fe3fef8b326c5595f0dcd7302e0825bbebd50'


def by_basename(hashes):
    result = {}
    for filename, value in hashes.items():
        name = Path(filename).name
        if name in result:
            raise ValueError('Ambiguous duplicate source/input basename')
        result[name] = value
    return result


def paired_contract(on, off):
    if on['protocol'].get('structure') is not True or off['protocol'].get('structure') is not False:
        raise ValueError('Expected explicitly structure-on and structure-off protocols')
    if on['arguments'].get('no_structure', False) or not off['arguments'].get('no_structure', False):
        raise ValueError('Structure flag disagrees with protocol')
    p1, p2 = dict(on['protocol']), dict(off['protocol'])
    p1.pop('structure'); p2.pop('structure')
    if p1 != p2:
        raise ValueError('Model/training protocol differs beyond structure')
    for name in ['split_seeds', 'model_seeds', 'folds', 'threads']:
        if on['arguments'][name] != off['arguments'][name]:
            raise ValueError(f'Paired argument differs: {name}')
    if on['arguments'].get('families', ['mlp', 'ddpm']) != off['arguments'].get('families', ['mlp', 'ddpm']):
        raise ValueError('Model families differ')
    if by_basename(on['input_sha256']) != by_basename(off['input_sha256']):
        raise ValueError('Input file hashes differ')
    if on['versions'] != off['versions']:
        raise ValueError('Execution library versions differ')
    a, b = by_basename(on['source_sha256']), by_basename(off['source_sha256'])
    current = core.digest(Path(core.__file__))
    if b.get('repeated_latent_validation.py') != current:
        raise ValueError('Off run source is not this reviewed ablation implementation')
    if a.get('repeated_latent_validation.py') not in {current, LEGACY_ON_RUNNER_SHA256}:
        raise ValueError('On run source is neither current nor the reviewed legacy on implementation')
    if a.get('first_structure_mlp.py') != b.get('first_structure_mlp.py') or b.get('first_structure_mlp.py') != core.digest(core.SNAPSHOT):
        raise ValueError('Data loader source differs')


def run_details(run, info, audit):
    import pandas as pd
    expected = [dict(split_seed=s, fold=f, model_seed=m, model=k)
                for s in info['arguments']['split_seeds'] for f in range(info['arguments']['folds'])
                for m in info['arguments']['model_seeds'] for k in info['arguments'].get('families', ['mlp', 'ddpm'])]
    done = json.loads((run / 'completed.json').read_text())
    if done.get('completed_jobs') != len(expected) or done.get('expected_jobs') != len(expected):
        raise ValueError('Incomplete run')
    frames = []
    for job in expected:
        task = run / f'split_{job["split_seed"]}' / f'fold_{job["fold"]}' / f'{job["model"]}_seed{job["model_seed"]}'
        record = core.read_completion(task, job)
        if record is None:
            raise ValueError('Missing valid completion record')
        file = (run / record['metrics_path']).resolve()
        if task.resolve() not in file.parents:
            raise ValueError('Metrics outside job folder')
        relative = str(file.relative_to(task.resolve()))
        if relative not in record['sha256']:
            raise ValueError('Metrics are not covered by completion hashes')
        frame = pd.read_csv(file)
        if frame.condition_id.duplicated().any():
            raise ValueError('Duplicate target in metrics')
        for name, value in job.items():
            frame[name] = value
        frames.append(frame)
        audit[str(file)] = core.digest(file)
        audit[str(task / 'complete.json')] = core.digest(task / 'complete.json')
    return pd.concat(frames, ignore_index=True)


def execute(args):
    import numpy as np
    import pandas as pd
    on_run, off_run, out = args.on_run.resolve(), args.off_run.resolve(), args.out.resolve()
    core.protect_output(out, [on_run, off_run])
    if out.exists():
        raise ValueError('Use a new comparison output folder')
    on = json.loads((on_run / 'run.json').read_text())
    off = json.loads((off_run / 'run.json').read_text())
    paired_contract(on, off)
    core.protect_output(out, [Path(off['arguments']['prepared']), Path(off['arguments']['structure'])])
    audit = {}
    # Exact split assignments and baseline predictions must match, including target order.
    for seed in on['arguments']['split_seeds']:
        for relative in [f'compound_folds_{seed}.csv'] + [f'split_{seed}/fold_{f}/compound_split.csv' for f in range(on['arguments']['folds'])]:
            a, b = on_run / relative, off_run / relative
            if a.read_bytes() != b.read_bytes():
                raise ValueError(f'Compound split differs: {relative}')
            audit[str(a)], audit[str(b)] = core.digest(a), core.digest(b)
        for fold in range(on['arguments']['folds']):
            relative = f'split_{seed}/fold_{fold}/baseline_predictions.npz'
            a, b = on_run / relative, off_run / relative
            with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
                if set(x.files) != set(y.files):
                    raise ValueError('Baseline artifact fields differ')
                for field in x.files:
                    np.testing.assert_array_equal(x[field], y[field])
            audit[str(a)], audit[str(b)] = core.digest(a), core.digest(b)
    on_rows, off_rows = run_details(on_run, on, audit), run_details(off_run, off, audit)
    keys = ['split_seed', 'fold', 'model_seed', 'model', 'condition_id', 'compound', 'time_h', 'level']
    paired = on_rows.merge(off_rows, on=keys, suffixes=('_on', '_off'), how='outer', validate='one_to_one', indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('On/off target coverage differs')
    if any(not np.isfinite(paired[c]).all() or (paired[c] < 0).any() for c in ['response_mse_on', 'response_mse_off']):
        raise ValueError('Invalid condition MSE')
    # Each model/seed must contain every target once over all folds.
    if paired.duplicated(['split_seed', 'model_seed', 'model', 'condition_id']).any():
        raise ValueError('Repeated OOF target')
    groups = paired.groupby(['split_seed', 'model_seed', 'model'])[['response_mse_on', 'response_mse_off']].mean().reset_index()
    groups['rmse_on'] = np.sqrt(groups.response_mse_on)
    groups['rmse_off'] = np.sqrt(groups.response_mse_off)
    groups['structure_on_minus_off'] = groups.rmse_on - groups.rmse_off
    groups['structure_on_improvement_pct'] = np.where(groups.rmse_off > 0, 100 * (1 - groups.rmse_on / groups.rmse_off), np.nan)
    out.mkdir(parents=True)
    paired.drop(columns='_merge').to_csv(out / 'paired_condition_metrics.csv', index=False)
    groups.to_csv(out / 'paired_by_split_model_seed.csv', index=False)
    summary = groups.groupby('model').agg(mean_rmse_on=('rmse_on', 'mean'), mean_rmse_off=('rmse_off', 'mean'),
        mean_on_minus_off=('structure_on_minus_off', 'mean'), std_on_minus_off=('structure_on_minus_off', 'std'),
        units_on_lower=('structure_on_minus_off', lambda x: int((x < 0).sum())), units=('structure_on_minus_off', 'size'))
    summary.to_csv(out / 'summary.csv')
    if args.bootstrap_samples:
        rows = []
        for model, frame in paired.groupby('model'):
            target = frame.groupby(['condition_id', 'compound'])[['response_mse_on', 'response_mse_off']].mean()
            sums = target.groupby(level='compound').sum()
            counts = target.groupby(level='compound').size().reindex(sums.index).to_numpy()
            draws = np.random.default_rng(81542).integers(len(sums), size=(args.bootstrap_samples, len(sums)))
            values = np.sqrt(sums.to_numpy()[draws].sum(1) / counts[draws].sum(1)[:, None])
            delta = values[:, 0] - values[:, 1]
            lo, hi = np.quantile(delta, [.025, .975])
            rows.append(dict(model=model, pooled_on_minus_off=float(np.sqrt(target.response_mse_on.mean()) - np.sqrt(target.response_mse_off.mean())),
                percentile_95_low=float(lo), percentile_95_high=float(hi), scope='compound_resampling_fixed_predictions_no_refitting'))
        pd.DataFrame(rows).to_csv(out / 'compound_bootstrap.csv', index=False)
    core.save_json(out / 'comparison.json', dict(on_run=str(on_run), off_run=str(off_run),
        on_run_sha256=core.digest(on_run/'run.json'), off_run_sha256=core.digest(off_run/'run.json'),
        comparator_sha256=core.digest(Path(__file__)), checked_artifacts=audit,
        inputs_protocol_versions_splits_and_baselines_match=True,
        compatibility='current_core_or_reviewed_9a71e16_on_source_with_current_off_source',
        model_deserialized=False, training_performed=False))
    core.save_json(out / 'completed.json', dict(compared_units=len(groups)))
    print('STRUCTURE COMPARISON COMPLETE:', out, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--on-run', type=Path, required=True)
    p.add_argument('--off-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    args = p.parse_args(argv)
    if args.bootstrap_samples < 0:
        p.error('Invalid bootstrap count')
    execute(args)


if __name__ == '__main__':
    main()
