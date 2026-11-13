"""Reanalyse the two Gemma-2-9B k-way jobs on CPU.

Usage: python reanalysis/kway.py [results-directory]

The GPU used torch.median to choose each set's threshold and normalised
mixture weights. Use the same lower median for the mixture response curve.
Intervals resample direction sets at fixed k, conditional on the anchors
and the finite direction pool. Sets at different k need not be nested.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze import fam, med_ci
from fsa.fit import intercept, kway_exponent


def summarise(job):
    with (job / "meta.json").open() as handle:
        meta = json.load(handle)
    summary = {}
    for k in meta["ks"]:
        radii = []
        midpoints = []
        names_seen = set()
        bounds = set()
        slots = Counter()
        files = sorted(job.glob(f"k{k:02d}_set*.npz"))
        for path in files:
            with np.load(path) as data:
                names = data["names"].tolist()
                names_seen.add(tuple(sorted(names)))
                slots.update(fam(name) for name in names)
                bounds.add(float(data["overlap_bound"]))
                alpha = data["alpha_i"]
                if "mix_normed" not in data or not np.isfinite(alpha).all():
                    continue
                curves = data["mix_normed"]
                middle = (len(curves) - 1) // 2
                lower = np.partition(curves, middle, axis=0)[middle]
                ordinary = np.median(curves, axis=0)
                scale = np.sqrt(k) / np.linalg.norm(np.sin(np.deg2rad(alpha)))
                for curve, output in [(lower, radii), (ordinary, midpoints)]:
                    t = intercept(data["angles_deg"], curve, float(data["thr"]))
                    if np.isfinite(t):
                        output.append(float(scale * np.sin(np.deg2rad(t))))
        row = dict(planned=meta["n_sets"], files=len(files), distinct_sets=len(names_seen),
                   crossed=len(radii), overlap_caps=sorted(bounds), direction_slots=dict(slots))
        if radii:
            r = med_ci(radii)
            row.update(radius=r, implied_p=[float(kway_exponent(k, x)) for x in r])
        if midpoints:
            row["midpoint_median_radius"] = float(np.median(midpoints))
        summary[k] = row
    if "radius" in summary.get(2, {}):
        r2 = summary[2]["radius"][0]
        slope = np.log2(r2)
        delta = r2 ** -2 - 1
        for k, row in summary.items():
            q = 1 + (k - 1) * delta
            row["lp_prediction_from_k2"] = float(k ** slope)
            row["quadratic_prediction_from_k2"] = float(q ** -0.5) if q > 0 else None
    return dict(job=job.name, has_done_marker=(job / "DONE").is_file(),
                anchors=meta["n_anchors"], summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=ROOT / "raw" / "results")
    args = parser.parse_args()
    print(json.dumps([summarise(args.results / f"g9_kway_{family}")
                      for family in ["contrastive", "randdiff"]], indent=2, allow_nan=False))
