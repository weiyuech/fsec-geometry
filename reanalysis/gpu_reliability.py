"""H3 reliability of paper-quadrant p on 90 pairs sharing a 300-anchor sample.

--cache optionally reuses per-pair CPU
fits as newly mirrored files arrive. Only a complete dataset produces a report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

AUDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT))
from analyze import fam
from fsa.fit import fit_quadrant

SEED = 20261112
N_BOOT = 200
N_SPLIT = 20
SIZES = (15, 30, 75, 150)


def common_draws(n):
    rng = np.random.default_rng(SEED)
    # Use identical anchor indices for all pairs, preserving anchor dependence.
    boots = rng.integers(n, size=(N_BOOT, n))
    perms = np.array([rng.permutation(n) for _ in range(N_SPLIT)])
    return boots, perms


def measure(path, ang, boots, perms):
    with np.load(path) as z:
        assert np.array_equal(z['angles_deg'], ang)
        g = z['l2']
    n = boots.shape[1]
    assert g.shape == (n, len(ang), len(ang)) and np.isfinite(g).all()

    def aggregate(indices):
        return fit_quadrant(ang, np.median(g[indices], axis=0))['p']

    splits = np.empty((len(SIZES), len(perms), 2))
    for i, size in enumerate(SIZES):
        for j, perm in enumerate(perms):
            splits[i, j] = [aggregate(perm[:size]), aggregate(perm[-size:])]
    return dict(full=aggregate(slice(None)), single=np.array([fit_quadrant(ang, gg)['p'] for gg in g]),
                boots=np.array([aggregate(idx) for idx in boots]), splits=splits)


def crossed(a):
    # Balanced two-way random-effects ANOVA of SINGLE-anchor p. Pair x anchor
    # interaction and measurement error are inseparable with one fit per cell.
    valid = np.isfinite(a).all(axis=0)
    a = a[:, valid]
    n_pair, n_anchor = a.shape
    if min(a.shape) < 2:
        return dict(complete_anchors=n_anchor, complete_pairs=n_pair)
    grand = a.mean()
    pair, anchor = a.mean(axis=1, keepdims=True), a.mean(axis=0, keepdims=True)
    ms_pair = n_anchor * np.var(pair, ddof=1)
    ms_anchor = n_pair * np.var(anchor, ddof=1)
    ms_error = ((a - pair - anchor + grand)**2).sum() / ((n_pair-1)*(n_anchor-1))
    return dict(complete_anchors=n_anchor, complete_pairs=n_pair,
                pair=float((ms_pair-ms_error)/n_anchor), anchor=float((ms_anchor-ms_error)/n_pair),
                interaction_plus_residual=float(ms_error))


def split_summary(splits):
    out = {}
    for size, values in zip(SIZES, splits.transpose(1, 2, 0, 3)):  # [size, split, pair, half]
        pearson, rank, abs_diff = [], [], []
        for x, y in values.transpose(0, 2, 1):
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            pearson.append(np.corrcoef(x, y)[0, 1])
            rank.append(spearmanr(x, y).statistic)
            abs_diff.append(np.median(abs(x-y)))
        r = float(np.median(pearson))
        out[str(size)] = dict(pearson=r, spearman=float(np.median(rank)),
                              median_absolute_difference=float(np.median(abs_diff)),
                              n_for_pearson_0_90=float(9*size*(1-r)/r) if r > 0 else None)
    return out


def summarize(data):
    full = np.array([r['full'] for r in data])
    boots = np.stack([r['boots'] for r in data])   # pair x common anchor resample
    assert np.isfinite(full).all() and np.isfinite(boots).all()
    family_medians = np.median(boots, axis=0)
    low, high = np.percentile(boots, [2.5, 97.5], axis=1)
    return dict(n=len(full), median_p=float(np.median(full)), above_2=int((full>2).sum()),
                median_p_anchor_ci=list(map(float,np.percentile(family_medians, [2.5,97.5]))),
                pair_cis_above_2=int((low>2).sum()), pair_cis_below_2=int((high<2).sum()),
                split=split_summary(np.stack([r['splits'] for r in data])),
                crossed_single_anchor_variances=crossed(np.stack([r['single'] for r in data])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=AUDIT/'raw/results/g9_reliability_300anchors')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('gpu_reliability.json'))
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    meta_path = args.results / 'meta.json'
    meta = json.loads(meta_path.read_text())
    n = meta['n_anchors']
    ang = np.array(meta['angles_deg'])
    assert n == 300 and len(meta['pairs']) == 90 and ang.min() == 0
    boots, perms = common_draws(n)
    # Cache keys include source code, job metadata and the raw input bytes.
    fingerprint = hashlib.sha256()
    for p in (Path(__file__), AUDIT/'fsa/fit.py', meta_path):
        fingerprint.update(p.read_bytes())
    if args.cache is not None:
        args.cache.mkdir(parents=True, exist_ok=True)
    results, pairs = [], []
    for a, b in meta['pairs']:
        path = args.results/f'{a}__{b}.npz'
        if not path.exists():
            continue
        state = fingerprint.copy()
        state.update(path.read_bytes())
        cache_file = args.cache/(state.hexdigest()+'.npz') if args.cache is not None else None
        if cache_file is not None and cache_file.exists():
            with np.load(cache_file) as cache:
                row = dict(cache)
        else:
            row = measure(path, ang, boots, perms)
            if cache_file is not None:
                np.savez_compressed(cache_file, **row)
        results.append(row)
        pairs.append((a, b))
    if len(results) != len(meta['pairs']):
        print(f'Fits available for {len(results)}/{len(meta["pairs"])} pairs. The complete report is pending.')
        return
    families = ['-'.join(sorted([fam(a),fam(b)])) for a,b in pairs]
    summaries = {f: summarize([r for group,r in zip(families,results) if group==f]) for f in sorted(set(families))}
    report = dict(
        n_anchors=n, unique_token_sequences=(len(set(map(tuple, meta['anchor_ids']))) if 'anchor_ids' in meta else meta['n_unique_anchor_sequences']),
        methods={
            'p':'Fit the positive-quadrant contour of the anchor-median response grid, using each sample’s paper threshold.',
            'bootstrap':'200 resamples of the 300 anchor indices, shared across all pairs. CIs for family medians condition on the selected pairs. Pair intervals are pointwise, with no multiplicity adjustment.',
            'split':'20 seeded draws of disjoint equal-sized anchor subsets; median Pearson and Spearman across draws for absolute and rank association over the same pairs. The median absolute error between halves is in exponent units.',
            'target':'Spearman–Brown extrapolation for Pearson reliability of the fitted aggregate, using 9 * n * (1-r) / r at each subset size n. This assumes similarly distributed independent anchors and inverse-n error variance. Nonlinearity of median-grid fits makes values approximate. Pooling families inflates between-pair dispersion.',
            'crossed':'Method-of-moments variance components for single-anchor p using all pairs and anchors with finite fits for all pairs in each stratum. Negative estimates are retained. Residual includes pair-by-anchor interaction. This is a different estimand from p fitted on an anchor-median grid.'
        }, all=summarize(results), families=summaries,
        rows=[dict(pair=pair,family=family,p=float(row['full']),
                   anchor_ci=list(map(float,np.percentile(row['boots'],[2.5,97.5]))),
                   finite_single_anchor_fits=int(np.isfinite(row['single']).sum()))
              for pair,family,row in zip(pairs,families,results)])
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False)+'\n')
    for name,r in [('ALL',report['all']), *summaries.items()]:
        print(name, r['median_p'], r['median_p_anchor_ci'], r['split']['150'])


if __name__ == '__main__':
    main()
