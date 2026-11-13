"""Sensitivity of the Gemma-2-9B language sign contrast to thresholds and shared anchors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze import LANG
from fsa.fit import fit_four_quadrants


def sign_difference(angles, grid, language_axis, fraction):
    q = fit_four_quadrants(angles, grid, frac=fraction)
    signs = ("++", "+-", "-+", "--")
    return np.mean([q[s]["p"] for s in signs if s[language_axis] == "+"]) - np.mean(
        [q[s]["p"] for s in signs if s[language_axis] == "-"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "raw/results/g9_contrastive_signed")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    folder = args.results
    meta = json.loads((folder / "meta.json").read_text())
    angles = np.array(meta["angles_deg"])
    n_anchors = meta["n_anchors"]
    # Every pair was measured on the same anchors. Preserve that dependence in each draw.
    draws = np.random.default_rng(719).integers(n_anchors, size=(250, n_anchors))
    fractions = (0.35, 0.50, 0.65)
    threshold_diffs = []
    bootstrap_diffs = []
    for a, b in meta["pairs"]:
        if (a in LANG) == (b in LANG):
            continue
        language_axis = 0 if a in LANG else 1
        with np.load(folder / f"{a}__{b}.npz") as z:
            l2 = z["l2"]
        median = np.median(l2, axis=0)
        threshold_diffs.append([sign_difference(angles, median, language_axis, f) for f in fractions])
        bootstrap_diffs.append([sign_difference(angles, np.median(l2[i], axis=0), language_axis, 0.5)
                                for i in draws])
    threshold_diffs = np.asarray(threshold_diffs)
    bootstrap_diffs = np.asarray(bootstrap_diffs)
    if not np.isfinite(bootstrap_diffs).all() or not np.isfinite(threshold_diffs).all():
        raise ValueError("At least one required contrast could not be fitted")
    output = {
        "n_pairs": len(threshold_diffs),
        "n_shared_anchors": n_anchors,
        "fraction_of_smallest_half_axis_maximum": {
            f: {"median_p_difference": float(np.median(v)), "positive_pairs": int((v > 0).sum())}
            for f, v in zip(fractions, threshold_diffs.T)
        },
        "anchor_bootstrap_draws": len(draws),
        "median_difference_anchor_bootstrap_95": np.percentile(
            np.median(bootstrap_diffs, axis=0), [2.5, 97.5]).tolist(),
        "positive_pair_count_anchor_bootstrap_95": np.percentile(
            (bootstrap_diffs > 0).sum(axis=0), [2.5, 97.5]).tolist(),
    }
    print(json.dumps(output, indent=2))
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
