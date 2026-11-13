"""Refit the signed pair experiment from complete grids."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

AUDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT))
from analyze import fam
from fsa import fit


def interval(values):
    values = np.array(values, dtype=float)
    assert np.all(np.isfinite(values))
    rng = np.random.default_rng(0)
    medians = np.median(rng.choice(values, (2000, len(values))), axis=1)
    return [float(np.median(values)), *map(float, np.percentile(medians, [2.5, 97.5]))]


def summary(rows):
    keys = ('p_paper', 'p_same', 'p_mixed', 'common', 'flip')
    return dict(n=len(rows), **{k: interval([r[k] for r in rows]) for k in keys},
                common_negative=sum(r['common'] < 0 for r in rows),
                flip_negative=sum(r['flip'] < 0 for r in rows))


def sign_split(rows, family):
    by_direction = defaultdict(list)
    for r in rows:
        names = r['pair']
        ix = 0 if fam(names[0]) == family else 1
        values = {}
        for key in ('p', 'delta'):
            for name, sign in [('plus', '+'), ('minus', '-')]:
                values[f'{key}_{name}'] = float(np.mean([
                    r['quadrants'][q][key] for q in ('++', '+-', '-+', '--') if q[ix] == sign
                ]))
            values[f'{key}_difference'] = values[f'{key}_plus'] - values[f'{key}_minus']
        by_direction[names[ix]].append(values)
    all_values = [v for group in by_direction.values() for v in group]
    return dict(n=len(rows), **{k: interval([v[k] for v in all_values]) for k in all_values[0]},
                count_p_plus_greater=sum(v['p_difference'] > 0 for v in all_values),
                by_direction={name: dict(n=len(group), median_p_difference=float(np.median([
                    v['p_difference'] for v in group]))) for name, group in sorted(by_direction.items())})


def analyze(path):
    meta = json.loads((path / 'meta.json').read_text())
    ang = np.array(meta['angles_deg'])
    rows = []
    for a, b in meta['pairs']:
        with np.load(path / f'{a}__{b}.npz') as z:
            assert np.array_equal(z['angles_deg'], ang)
            grids = z['l2']
            assert grids.shape == (meta['n_anchors'], len(ang), len(ang))
            median = np.median(grids, axis=0)
        result = fit.fit_four_quadrants(ang, median)
        ppaper = fit.fit_quadrant(*fit.quadrant(ang, median, 1, 1))['p']
        rows.append(dict(pair=[a, b], family='-'.join(sorted([fam(a), fam(b)])), p_paper=ppaper,
                         **{k: result[k] for k in ('p_same', 'p_mixed', 'common', 'flip')},
                         quadrants={q: {k: result[q][k] for k in ('p', 'delta')}
                                    for q in ('++', '+-', '-+', '--')}))
    families = Counter(r['family'] for r in rows)
    family_rows = {k: [r for r in rows if r['family'] == k] for k in families}
    signs = {k: sign_split(family_rows[k + '-sem'], k) for k in ('lang', 'code') if k + '-sem' in families}
    return dict(n_anchors=meta['n_anchors'], all=summary(rows),
                families={k: summary(v) for k, v in family_rows.items()}, sign_split=signs, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=AUDIT/'raw/results')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('gpu_shape.json'))
    args = parser.parse_args()
    jobs = ('g9_contrastive_signed', 'g9_randdiff_signed')
    report = {job: analyze(args.results/job) for job in jobs}
    report['methods'] = {
        'four_quadrant': 'Common threshold across signs; paper ++ is fit separately at its original threshold.',
        'triples': 'Median, 2.5th and 97.5th percentiles of 2000 pair-bootstrap medians. These intervals condition on this anchor sample and ignore dependencies from shared directions.',
        'sign_split': 'Within each pair, average over both semantic signs before taking the median over pairs. Difference is computed within pair before taking medians. Positive language/code sign is the projection of target minus English prompt directions.',
    }
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    for job in jobs:
        print(job, report[job]['all'])


if __name__ == '__main__':
    main()
