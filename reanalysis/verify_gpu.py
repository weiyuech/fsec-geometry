"""Check the shapes, finite values, pair identities, and mixture sets in a fresh queue."""
import argparse
from itertools import product
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fsa.data import low_overlap_pairs, low_overlap_sets
from toy_sweep import ANGLES, DIMS, KS, NOISES, RANGES, SEEDS, key


def check(condition, message):
    if not condition:
        raise ValueError(message)


def same_names(actual, expected):
    actual, expected = set(actual), set(expected)
    check(actual == expected, f"Missing {sorted(expected-actual)[:4]}, unexpected {sorted(actual-expected)[:4]}")


def crc(path):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        check(bad is None, f"{path}: CRC failed for {bad}")


def verify_toy(path, job):
    cells = product(job.get("ds", DIMS), job.get("seeds", SEEDS),
                    job.get("anchor_noises", NOISES), job.get("amaxes", RANGES))
    cell_names = {f"tanh3_d{d}_s{s}_an{n}_amax{a}.json": (d,s,n,a) for d,s,n,a in cells}
    same_names((p.name for p in path.glob("*_d*_s*_an*_amax*.json")), cell_names)
    theta = job.get("thetas_deg", ANGLES)
    rows = []
    for filename, cell in cell_names.items():
        expected = {(*cell,t,"pair",0) for t in theta}
        expected.update((*cell,t,"kway",k) for t,k in product(theta, job.get("ks", KS)))
        parsed = json.loads((path / filename).read_text())
        same_names(map(key, parsed), expected)
        check(len(parsed) == len(expected), f"Duplicate records in {filename}")
        for row in parsed:
            check(row["act"] == "tanh3" and row["H"] == 1024, f"Unexpected model in {filename}")
            if row["kind"] == "pair":
                check(0 < row["n_fit"] <= job["n_sets"], f"Invalid fit count in {filename}")
        rows.extend(parsed)
    saved = json.loads((path / "summary.json").read_text())
    canonical = lambda data: json.dumps(sorted(data, key=key), sort_keys=True)
    check(canonical(rows) == canonical(saved), "Toy summary differs from individual cells")
    return dict(cells=len(cell_names), records=len(rows),
                plus_plus_fits=sum(r.get("n_fit", 0) for r in rows))


def verify_arrays(path, job):
    meta = json.loads((path / "meta.json").read_text())
    n = job["n_anchors"]
    check(meta["family"] == job["family"] and meta["n_anchors"] == n, "Metadata mismatch")
    check(meta["L"] == job["perturb_layer"], "Perturbation layer mismatch")
    step = job["step_deg"]
    amax = job.get("max_deg", 60)
    start = -amax if job["exp"] == "grid" and job["signed"] else 0
    angles = np.arange(start, amax + 0.5 * step, step, dtype=float)
    check(np.array_equal(meta["angles_deg"], angles), "Metadata angle axis differs from specification")
    g = len(angles)
    descriptors = {}
    if job["exp"] == "grid":
        weights = path / "directions.pt"
        crc(weights)
        if "pairs" in job:
            pairs = job["pairs"]
        else:
            directions = torch.load(weights, map_location="cpu", weights_only=True)
            pairs = low_overlap_pairs(directions, limit=job["limit"], seed=job.get("dir_seed", 0))
        check(list(map(tuple, pairs)) == list(map(tuple, meta["pairs"])), "Manifest pairs differ")
        anchors = np.asarray(meta["anchor_ids"])
        check(anchors.shape == (n,5), "Unexpected anchor shape")
        descriptors = {f"{a}__{b}.npz": None for a,b in pairs}
        check(len(descriptors) == len(pairs), "Repeated pair in specification")
    else:
        ks = job.get("ks", [2, 3, 4, 6, 8, 12, 16])
        check(meta["ks"] == ks and meta["n_sets"] == job["n_sets"], "Mixture design differs")
        source = path.parent / f"g9_{job['family']}_signed" / "directions.pt"
        dirs = torch.load(source, map_location="cpu", weights_only=True)
        for k in ks:
            sets, bound = low_overlap_sets(dirs, k, job["n_sets"], seed=job.get("dir_seed", 0) + k)
            check(len(sets) == job["n_sets"], f"Unable to reconstruct sets of size {k}")
            for i, names in enumerate(sets):
                descriptors[f"k{k:02d}_set{i:03d}.npz"] = (names, bound)
    same_names((p.name for p in path.glob("*.npz")), descriptors)
    for filename, descriptor in descriptors.items():
        target = path / filename
        crc(target)
        with np.load(target, allow_pickle=False) as z:
            check(np.array_equal(z["angles_deg"], angles), f"Wrong angles in {filename}")
            shapes = dict(angles_deg=(g,), l2=(n,g,g), cos12=())
            if descriptor is not None:
                names, bound = descriptor
                k = len(names)
                check(np.array_equal(names, z["names"]) and bound == float(z["overlap_bound"]),
                      f"Direction set differs in {filename}")
                shapes = dict(angles_deg=(g,), names=(k,), axis=(n,k,g), alpha_i=(k,), thr=(),
                              overlap_bound=(), mix_normed=(n,g), mix_equal=(n,g),
                              w_normed=(k,), w_equal=(k,))
            same_names(z.files, shapes)
            for name, shape in shapes.items():
                a = z[name]
                check(a.shape == shape, f"Wrong shape for {name} in {filename}")
                if a.dtype.kind in "fiu":
                    check(np.isfinite(a).all(), f"Nonfinite values for {name} in {filename}")
    return dict(npz_crc_checked=len(descriptors))


def queue(results, spec):
    specs = json.loads(spec.read_text())
    jobs = [{**specs["defaults"], **j} for j in specs["jobs"]]
    statefile = results / "queue_status.json"
    states = json.loads(statefile.read_text()) if statefile.exists() else {}
    names = {j["name"] for j in jobs}
    report = {}
    for job in jobs:
        name = job["name"]
        path = results / name
        row = report[name] = dict(status=states.get(name), verified=False)
        if not (path / "DONE").exists():
            continue
        try:
            check(str(states.get(name)).startswith("done"), "DONE marker disagrees with queue status")
            row.update(verify_toy(path, job) if job["exp"] == "toy" else verify_arrays(path, job))
            row["verified"] = True
        except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as exc:
            row["error"] = str(exc)
    return dict(verified=all(r["verified"] for r in report.values()) and set(states) == names,
                unexpected_status_names=sorted(set(states) - names), jobs=report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "raw/results")
    ap.add_argument("--jobs", type=Path, default=ROOT / "jobs/main.json")
    args = ap.parse_args()
    report = queue(args.results, args.jobs)
    print(json.dumps(report, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
