"""Compare signed grids for the matched H4 panel. Run again after each mirror update."""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze import fam
from fsa import fit

RESULTS = ROOT / "raw/results"
OUT = Path(__file__).with_suffix(".json")
REFERENCE = "g9_contrastive_signed"
METRICS = ("p_paper", "p_pp", "p_pm", "p_mp", "p_mm", "p_same", "p_mixed",
           "p_axis2_positive", "p_axis2_negative", "p_axis2_positive_minus_negative", "flip", "common")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def finite_median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def fit_pair(path, angles, n_anchors):
    a, b = path.stem.split("__")
    with np.load(path) as z:
        response = z["l2"]
        if response.shape != (n_anchors, len(angles), len(angles)) or not np.array_equal(angles, z["angles_deg"]):
            raise ValueError(f"Unexpected grid dimensions or angles in {path}")
        if not np.isfinite(response).all() or not np.isfinite(z["cos12"]).all():
            raise ValueError(f"Nonfinite data in {path}")
        cosine = float(z["cos12"])
        median_grid = np.median(response, axis=0)
    angles_pp, grid_pp = fit.quadrant(angles, median_grid)
    paper = fit.fit_quadrant(angles_pp, grid_pp)
    shared = fit.fit_four_quadrants(angles, median_grid)
    row = dict(family="-".join(sorted((fam(a), fam(b)))), cos12=cosine, p_paper=paper["p"],
               thr_paper=paper["thr"], thr_shared=shared["thr"])
    for key, quadrant in (("p_pp", "++"), ("p_pm", "+-"), ("p_mp", "-+"), ("p_mm", "--")):
        row[key] = shared[quadrant]["p"]
    row["p_axis2_positive"] = (row["p_pp"] + row["p_mp"]) / 2
    row["p_axis2_negative"] = (row["p_pm"] + row["p_mm"]) / 2
    row["p_axis2_positive_minus_negative"] = row["p_axis2_positive"] - row["p_axis2_negative"]
    row.update({key: shared[key] for key in ("p_same", "p_mixed", "flip", "common")})
    return {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in row.items()}


def summarize(rows, direction_sensitivity=False):
    groups = {"ALL": rows}
    for key, row in rows.items():
        groups.setdefault(row["family"], {})[key] = row
    summary = {}
    for family, selected in groups.items():
        group = list(selected.values())
        medians = {k: finite_median(r[k] for r in group) for k in METRICS}
        n_valid = {k: sum(r[k] is not None for r in group) for k in METRICS}
        summary[family] = dict(n_pairs=len(group), valid=n_valid, median=medians)
        if direction_sensitivity:
            # Each omitted direction removes all pairs containing it in either position.
            directions = {d for p in selected for d in p.split("__")}
            ranges = {}
            for metric in ("p_paper", "common", "flip", "p_axis2_positive_minus_negative"):
                leave_out = [finite_median(row[metric] for p, row in selected.items() if d not in p.split("__"))
                             for d in sorted(directions)]
                valid = [v for v in leave_out if v is not None]
                ranges[metric] = [min(valid), max(valid)] if valid else None
            summary[family]["leave_one_direction_out_range"] = ranges
    return summary


def analyze_job(job, panel, cached, revision):
    name = job["name"]
    directory = RESULTS / name
    if not all((directory / f).is_file() for f in ("DONE", "meta.json")):
        return None
    meta = json.loads((directory / "meta.json").read_text())
    all_pairs = ["__".join(pair) for pair in job["pairs"]]
    if meta["pairs"] != job["pairs"] or len(set(all_pairs)) != len(all_pairs):
        raise ValueError(f"{name}: recorded pair identities/order differ from job specification")
    actual = {path.stem for path in directory.glob("*__*.npz") if not path.name.endswith(".tmp.npz")}
    if actual - set(all_pairs):
        raise ValueError(f"{name}: unexpected pair files {sorted(actual - set(all_pairs))}")
    if set(all_pairs) != actual:
        return None  # The mirror can copy DONE before its remaining grids.
    paths = [directory / f"{k}.npz" for k in panel]
    files = paths + [directory / "meta.json"]
    if (directory / "directions.pt").exists():
        files.append(directory / "directions.pt")
    signature = digest([revision, [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files]])
    if cached and cached["input_signature"] == signature:
        return cached
    ang = np.asarray(meta["angles_deg"])
    expected = np.arange(-job["max_deg"], job["max_deg"] + job["step_deg"] / 2, job["step_deg"])
    n = meta["n_anchors"]
    if "anchor_ids" in meta:
        anchor_shape = np.shape(meta["anchor_ids"])
        anchor_digest = digest(meta["anchor_ids"])
    else:
        anchor_shape = (n, meta["anchor_seq_len"])
        anchor_digest = meta["anchor_ids_sha256"]
    if not np.array_equal(ang, expected) or anchor_shape != (job["n_anchors"], 5):
        raise ValueError(f"{name}: angles or anchor count differ from specification")
    rows = {p.stem: fit_pair(p, ang, n) for p in paths}
    summary = summarize(rows)
    for family, group in summary.items():
        members = [r for r in rows.values() if family == "ALL" or r["family"] == family]
        cos = np.abs([r["cos12"] for r in members])
        group.update(n_abs_cos_lt_point1=int(sum(cos < 0.1)), median_abs_cos=float(np.median(cos)),
                     max_abs_cos=float(np.max(cos)))
        for metric in ("p_paper", "p_pp"):
            group[f"n_{metric}_gt_2"] = sum(r[metric] is not None and r[metric] > 2 for r in members)
    fields = ("L", "M", "path", "n_anchors", "anchor_seed", "anchor_skip")
    return dict(input_signature=signature, job=name, model=job["model"], shuffle_seed=job.get("shuffle_seed"),
                job_pair_count=len(all_pairs), anchor_ids_sha256=anchor_digest,
                dir_prompts_sha256=digest(meta["dir_meta"]), angles_sha256=digest(meta["angles_deg"]),
                **{k: meta[k] for k in fields}, summary=summary, rows=rows)


def directions(job):
    import torch

    vectors = torch.load(RESULTS / job["job"] / "directions.pt", map_location="cpu", weights_only=True)
    return {n: v.detach().double().numpy() for n, v in vectors.items()}


def direction_alignment(trained, control):
    if not all((RESULTS / j["job"] / "directions.pt").is_file() for j in (trained, control)):
        stored = json.loads((RESULTS / "direction_alignment.json").read_text())
        return stored["trained_minus_" + control["job"]]
    train_vectors, control_vectors = directions(trained), directions(control)
    names = {d for pair in trained["rows"] for d in pair.split("__")}
    individual = {}
    for n in sorted(names):
        a, b = train_vectors[n], control_vectors[n]
        cosine = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if not np.isfinite(cosine):
            raise ValueError(f"Invalid same-label direction cosine for {n}")
        individual[n] = cosine
    grouped = {family: finite_median(c for n, c in individual.items() if fam(n) == family)
               for family in sorted({fam(n) for n in names})}
    return dict(median_by_direction_family=grouped, signed_cosine_by_direction=individual)


def paired(trained, controls):
    matched_fields = ("L", "M", "n_anchors", "anchor_ids_sha256", "dir_prompts_sha256", "angles_sha256")
    for other in controls:
        if trained["rows"].keys() != other["rows"].keys() or any(trained[k] != other[k] for k in matched_fields):
            raise ValueError(f"Unmatched training/control inputs for {trained['model']}")
    diffs, low_cos = {}, {}
    for pair, reference in trained["rows"].items():
        nulls = [other["rows"][pair] for other in controls]
        row = {"family": reference["family"]}
        for k in METRICS:
            values = [r[k] for r in nulls]
            row[k] = (reference[k] - float(np.mean(values))
                      if reference[k] is not None and None not in values else None)
        diffs[pair] = row
        if all(abs(r["cos12"]) < 0.1 for r in [reference, *nulls]):
            low_cos[pair] = row
    result = dict(trained_minus_control=summarize(diffs, direction_sensitivity=True),
                  both_abs_cos_lt_point1=summarize(low_cos), rows=diffs)
    if len(controls) == 1:
        result["same_label_direction_alignment"] = direction_alignment(trained, controls[0])
    return result


def main():
    global RESULTS, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    RESULTS, OUT = args.results, args.out
    spec = json.loads((ROOT / "jobs/main.json").read_text())
    jobs = {j["name"]: {**spec["defaults"], **j} for j in spec["jobs"]
            if j["name"] == REFERENCE or j["name"].startswith(("null_", "g9_shuffled"))}
    panel = ["__".join(p) for p in jobs["g9_shuffled0_signed"]["pairs"]]
    if len(panel) != len(set(panel)) or not set(panel) <= {"__".join(p) for p in jobs[REFERENCE]["pairs"]}:
        raise ValueError("H4 panel must consist of unique ordered pairs from the unshuffled Gemma job")
    if any(set(panel) != {"__".join(p) for p in j["pairs"]} for name, j in jobs.items() if name != REFERENCE):
        raise ValueError("H4 controls use different pair panels")
    cached = json.loads(OUT.read_text())["jobs"] if OUT.exists() else {}
    revision = digest([hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), Path(fit.__file__), ROOT / "analyze.py")])
    complete = {}
    for name, job in jobs.items():
        result = analyze_job(job, panel, cached.get(name), revision)
        if result is not None:
            complete[name] = result
    comparisons = {}
    shuffles = [n for n, j in jobs.items() if "shuffle_seed" in j]
    for name in shuffles:
        if name in complete and REFERENCE in complete:
            comparisons[f"trained_minus_{name}"] = paired(complete[REFERENCE], [complete[name]])
    if REFERENCE in complete and all(n in complete for n in shuffles):
        comparisons["trained_minus_shuffle_mean"] = paired(complete[REFERENCE], [complete[n] for n in shuffles])
    for name in jobs:
        if "-step0" in name and name in complete and name.replace("-step0", "") in complete:
            comparisons[f"trained_minus_{name}"] = paired(complete[name.replace("-step0", "")], [complete[name]])
    out = dict(design={"pairs": panel, "family_counts": dict(Counter(
        "-".join(sorted(fam(n) for n in p.split("__"))) for p in panel)),
        "thresholds": "p_paper uses 0.5 times the smaller ++ axis maximum. Other fits use 0.5 times the smallest maximum across four half-axes, within each model/pair.",
        "pairing": "Same ordered names, angles, prompt checksums, anchor token IDs and extraction layers verified within each trained/control comparison. Direction vectors recomputed for each checkpoint. Differences are computed per pair before taking family medians. The three-shuffle comparison averages the three matched seed values for each pair.",
        "axis2_sign": "Shared-threshold p averaged over either sign of axis 1 within each pair. Axis 2 is the non-English language or code direction in every corresponding family pair. The sign gap is computed before taking medians.",
        "shuffle": "Each transformer block's 2D weight matrices are permuted entrywise within each matrix. The tokenizer, embeddings, and 1D normalization weights are preserved. Directions and anchor activations are re-extracted for each seed. Pythia step0 is a separate initialization control.",
        "cosine_sensitivity": "The prespecified panel is fixed across models. both_abs_cos_lt_point1 restricts to pairs with absolute pre-orthogonalization direction cosine below 0.1 in all checkpoints in the contrast.",
        "inference": "Descriptive fixed-panel medians. Pairs share directions and anchors; this summary treats neither pairs nor the three seeds as independent replicates of training."},
        pending=[n for n in jobs if n not in complete], jobs=complete, comparisons=comparisons)
    tmp = OUT.with_name(f".{OUT.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, OUT)
    for name, result in complete.items():
        s = result["summary"]["ALL"]
        print(name, s["n_pairs"], {k: s["median"][k] for k in ("p_paper", "p_pp", "common", "flip")})
    print(f"{len(complete)}/{len(jobs)} models/seeds analyzed.")


if __name__ == "__main__":
    main()
