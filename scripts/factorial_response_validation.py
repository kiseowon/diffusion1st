"""Four input arms x three model families; default is plan-only, never downloads data."""
import argparse
import json
from pathlib import Path
import repeated_latent_validation as core

ARMS = {'basic': (False, False), 'structure': (True, False),
        'pathway': (False, True), 'both': (True, True)}


def read_projection(prepared, kegg):
    import numpy as np
    import pandas as pd
    probes = pd.read_csv(prepared / 'probes.csv').probe.astype(str).tolist()
    provenance = json.loads((kegg / 'provenance.json').read_text())
    if list(map(str, provenance.get('probe_order', []))) != probes:
        raise ValueError('KEGG provenance probe_order must exactly match prepared/probes.csv')
    projection = np.load(kegg / 'projection.npy', allow_pickle=False).astype(float)
    if (projection.ndim != 2 or projection.shape[0] != len(probes) or
            not np.isfinite(projection).all() or (projection < 0).any()):
        raise ValueError('Invalid KEGG projection')
    if not np.allclose(projection.sum(0), 1., atol=1e-5, rtol=0):
        raise ValueError('KEGG pathway mean weights must sum to one per column')
    return projection


def augment(pack, data, ids, fit, projection, enabled):
    """Project standardized source deltas, average sources per target, train-fit scaler."""
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler
    source = pack['state']['input_preprocess']['source_scale'].transform(data.source)
    scores = source @ projection
    P = np.asarray([scores[data.source_ids[data.target_ids == cid]].mean(0) for cid in ids])
    scaler = StandardScaler().fit(P[fit])
    values = scaler.transform(P).astype(np.float32) if enabled else np.zeros_like(P, dtype=np.float32)
    pack['X'] = torch.cat([pack['X'], torch.tensor(values)], dim=1)
    pack['state'].update(pathway_scaler=scaler, pathway_projection=projection,
                         pathway_enabled=enabled, pathway_representation='fixed_weighted_mean')
    return pack


def summarize(out):
    import numpy as np
    import pandas as pd
    frames = []
    for arm in ARMS:
        frame = pd.read_csv(out / arm / 'oof_condition_metrics.csv')
        frame['arm'] = arm
        frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(out / 'factorial_condition_metrics.csv', index=False)
    units = detail.groupby(['arm', 'split_seed', 'model_seed', 'model']).response_mse.mean().pow(.5).rename('response_rmse').reset_index()
    units.to_csv(out / 'factorial_by_seed.csv', index=False)
    # Mean squared errors across model seeds, NOT an ensemble of predictions.
    summary = detail.groupby(['arm', 'model']).response_mse.mean().pow(.5).rename('pooled_response_rmse').reset_index()
    summary.to_csv(out / 'factorial_summary.csv', index=False)
    pivot = units[units.model_seed.ge(0)].pivot(index=['split_seed', 'model_seed', 'model'], columns='arm', values='response_rmse')
    comparisons = [('structure', 'basic'), ('pathway', 'basic'), ('both', 'basic'),
                   ('both', 'structure'), ('both', 'pathway')]
    paired = pivot.copy()
    for a, b in comparisons:
        paired[a + '_minus_' + b] = pivot[a] - pivot[b]
    paired['interaction_rmse'] = pivot.both - pivot.structure - pivot.pathway + pivot.basic
    paired.to_csv(out / 'factorial_paired_differences.csv')
    print(summary.to_string(index=False), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['prepared', 'structure', 'kegg', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--families', nargs='+', choices=['mlp', 'cgan', 'ddpm'], default=['mlp', 'cgan', 'ddpm'])
    p.add_argument('--model-seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    n = 4 * len(a.families) * len(a.model_seeds) * a.folds
    print(json.dumps(dict(mode='execute' if a.execute else 'plan_only', arms=ARMS,
                         families=a.families, neural_jobs=n, neural_fits_up_to=2*n,
                         prediction='direct_response_no_ridge', pathway='fixed_KEGG_weighted_mean',
                         note='All arms retain in vitro expression and experimental conditions'), indent=2), flush=True)
    # Reuse core CLI validation even in plan mode; no input IO or output creation.
    base = ['--prepared', str(a.prepared), '--structure', str(a.structure),
            '--split-seeds', str(a.split_seed), '--model-seeds', *map(str, a.model_seeds),
            '--families', *a.families, '--folds', str(a.folds), '--threads', str(a.threads),
            '--bootstrap-samples', '0', '--prediction-target', 'direct_response']
    args = core.parser().parse_args(base + ['--out', str(a.out)])
    if len(set(a.families)) != len(a.families) or len(set(a.model_seeds)) != len(a.model_seeds):
        p.error('Duplicate families or model seeds')
    if a.folds < 2 or a.threads < 1 or any(s < 0 or s >= 2**32 for s in [a.split_seed] + a.model_seeds):
        p.error('Invalid folds, threads or seeds')
    if not a.execute:
        return n
    projection = read_projection(a.prepared, a.kegg)
    original = core.prepare_stage
    protocol = core.PROTOCOL.copy()
    old_sources = getattr(core, 'EXTRA_SOURCE_FILES', [])
    old_inputs = getattr(core, 'EXTRA_INPUT_FILES', [])
    try:
        core.EXTRA_SOURCE_FILES = [Path(__file__)]
        core.EXTRA_INPUT_FILES = [a.kegg / 'projection.npy', a.kegg / 'provenance.json']
        for arm, (structure, pathway) in ARMS.items():
            core.PROTOCOL.update(pathway_representation='fixed_KEGG_weighted_mean',
                                 pathway_enabled=pathway, factorial_arm=arm)
            def prepare(*pos, _enabled=pathway, **kwargs):
                pack = original(*pos, **kwargs)
                return augment(pack, pos[1], pos[2], pos[5], projection, _enabled)
            core.prepare_stage = prepare
            args.out = a.out / arm
            args.no_structure = not structure
            core.execute(args)
        summarize(a.out)
    finally:
        core.prepare_stage = original
        core.PROTOCOL.clear(); core.PROTOCOL.update(protocol)
        core.EXTRA_SOURCE_FILES = old_sources
        core.EXTRA_INPUT_FILES = old_inputs
    return n


if __name__ == '__main__':
    main()
