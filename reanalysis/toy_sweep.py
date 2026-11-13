"""Summarise the saved toy_noise_range cells, using one observation per seed.

Run: python reanalysis/toy_sweep.py [result-directory] [--require-complete]
Only the Python standard library is required. Partial mirrors are supported.
"""
import argparse
from collections import Counter, defaultdict
from itertools import product
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DIMS = (4096, 8192, 16384)
SEEDS = range(5)
NOISES = (0.0, 0.0433, 0.0866, 0.1732)
RANGES = (1.5, 3.0, 8.0, 25.0)
ANGLES = (0, 15, 30, 45, 60, 90)
KS = (2, 3, 4, 6, 8, 12, 16, 24, 32)
PAIR_FITS = 30
CELL_FIELDS = ("d", "seed", "anchor_noise", "amax")


def key(row):
    return (*(row[k] for k in CELL_FIELDS), row["theta"], row["kind"], row.get("k", 0))


def summary(values):
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"n": 0}
    return dict(n=len(finite), median=median(finite), min=min(finite), max=max(finite))


def grouped(rows, by, measures):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in by)].append(row)
    return [
        dict(zip(by, keys), **{m: summary(r[m] for r in group) for m in measures})
        for keys, group in sorted(groups.items())
    ]


def analyse(directory):
    expected = {f"tanh3_d{d}_s{s}_an{n}_amax{a}.json": (d, s, n, a)
                for d, s, n, a in product(DIMS, SEEDS, NOISES, RANGES)}
    permitted = {(th, "pair", 0) for th in ANGLES}
    permitted.update((th, "kway", k) for th, k in product(ANGLES, KS))
    rows = {}
    files = sorted(directory.glob("*_d*_s*_an*_amax*.json"))
    for path in files:
        cell = expected[path.name]
        for row in json.loads(path.read_text()):
            ident = key(row)
            if ident[:4] != cell or ident[4:] not in permitted or row["act"] != "tanh3" or row["H"] != 1024:
                raise ValueError(f"Unexpected record {ident} in {path}")
            if ident in rows:
                raise ValueError(f"Duplicate record {ident}")
            if row["kind"] == "pair" and not 0 < row["n_fit"] <= PAIR_FITS:
                raise ValueError(f"Invalid pair fit count at {ident}")
            rows[ident] = row

    snapshot = directory / "summary.json"
    summary_matches = None
    if snapshot.exists():
        saved = json.loads(snapshot.read_text())
        ordered = sorted(rows.values(), key=key)
        summary_matches = json.dumps(sorted(saved, key=key), sort_keys=True) == json.dumps(ordered, sort_keys=True)

    pairs = [r for r in rows.values() if r["kind"] == "pair"]
    kway = [r for r in rows.values() if r["kind"] == "kway"]
    direction_fit = [r for r in pairs if r["theta"] == 0]
    extremes = [r for r in pairs if r["amax"] == max(RANGES) and r["anchor_noise"] in (0, max(NOISES))]
    k_extremes = [r for r in kway if r["theta"] == 0 and r["amax"] == max(RANGES)
                  and r["anchor_noise"] in (0, max(NOISES))]
    contrasts = []
    for amax in RANGES:
        for d in DIMS:
            differences = []
            crossing = 0
            for seed in SEEDS:
                clean = rows.get((d, seed, 0, amax, 0, "pair", 0))
                noisy = rows.get((d, seed, max(NOISES), amax, 0, "pair", 0))
                if clean and noisy:
                    differences.append(clean["p_median"] - noisy["p_median"])
                    crossing += clean["p_median"] > 2 > noisy["p_median"]
            if differences:
                contrasts.append(dict(d=d, amax=amax, clean_minus_noisy=summary(differences),
                                      n_smaller_with_noise=sum(v > 0 for v in differences),
                                      n_cross_from_above_to_below_2=crossing))
    missing = sorted(set(expected) - {p.name for p in files})
    coverage = dict(cells=len(files), expected_cells=len(expected),
                    missing_cells=missing, done_marker=(directory / "DONE").exists(),
                    saved_summary_matches_cells=summary_matches,
                    pair_rows=len(pairs), expected_pair_rows=len(expected) * len(ANGLES),
                    finite_plus_plus_fits=sum(r["n_fit"] for r in pairs),
                    expected_plus_plus_fits=len(expected) * len(ANGLES) * PAIR_FITS,
                    mixed_flip_common_finite_rows={k: sum(math.isfinite(r[k]) for r in pairs)
                                                   for k in ("p_mixed_median", "flip_median", "common_median")},
                    kway_rows=len(kway), expected_kway_rows=len(expected) * len(ANGLES) * len(KS),
                    finite_kway_radius_rows=sum(math.isfinite(r["norm_radius"]) for r in kway),
                    finite_kway_exponent_rows=sum(math.isfinite(r["p_implied"]) for r in kway),
                    fits_per_pair_row=dict(Counter(r["n_fit"] for r in pairs)))
    return dict(coverage=coverage,
                by_noise_range_size=grouped(direction_fit, ["d", "anchor_noise", "amax"], ["p_median"]),
                seed_matched_contrasts=contrasts,
                rotation_and_sign=grouped(extremes, ["d", "anchor_noise", "theta"],
                                          ["p_median", "p_mixed_median", "flip_median", "common_median"]),
                kway=grouped(k_extremes, ["d", "anchor_noise", "k"], ["norm_radius", "p_implied"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path, nargs="?", default=ROOT / "raw/results/toy_noise_range")
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    result = analyse(args.directory)
    cov = result["coverage"]
    if args.require_complete and (cov["missing_cells"] or not cov["done_marker"] or not cov["saved_summary_matches_cells"]):
        ap.error("Missing cell files, DONE marker, or matching summary")
    print(json.dumps(result, indent=2, allow_nan=False))
