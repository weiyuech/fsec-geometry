"""CPU analysis of results/<job>/ directories written by run_queue.py.

    python analyze.py results/<job> [results/<job2> ...]

grid jobs  -> per-pair fits on the anchor-median grid:
              p in the paper's quadrant (++), p in same-sign vs mixed-sign quadrants,
              cross-term flip and common part, plus anchor-bootstrap intervals.
kway jobs  -> normalised boundary radius and implied exponent per k.
Writes <job>/analysis.json and prints a summary table.
"""
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsa import fit

LANG = {"Portuguese", "Chinese", "Spanish", "French", "German", "Dutch", "Italian", "Japanese", "Russian", "Arabic"}
CODE = {"Python", "JavaScript", "Java", "Rust", "Haskell", "Cpp", "Go", "Lisp", "TypeScript"}


def fam(n):
    return "lang" if n in LANG else "code" if n in CODE else "ctrl" if n.startswith(("rdiff", "rand")) else "sem"


def med_ci(v, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [np.nan, np.nan, np.nan]
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(v, (n_boot, len(v))), axis=1)
    return [float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def analyze_grid(d, n_boot=40):
    meta = json.load(open(os.path.join(d, "meta.json")))
    ang = np.array(meta["angles_deg"], float)
    signed = ang.min() < 0
    rows = []
    rng = np.random.default_rng(0)
    for f in sorted(glob.glob(os.path.join(d, "*__*.npz"))):
        z = np.load(f)
        a, b = os.path.basename(f)[:-4].split("__")
        l2 = z["l2"]
        g = np.median(l2, axis=0)
        row = dict(a=a, b=b, family="-".join(sorted([fam(a), fam(b)])), cos12=float(z["cos12"]))
        if signed:
            fq = fit.fit_four_quadrants(ang, g)
            pa, pg = fit.quadrant(ang, g, 1, 1)
            row.update(p_paper=fit.fit_quadrant(pa, pg)["p"],      # paper's own threshold on the ++ quadrant
                       p_pp=fq["++"]["p"], p_pm=fq["+-"]["p"], p_mp=fq["-+"]["p"], p_mm=fq["--"]["p"],
                       p_same=fq["p_same"], p_mixed=fq["p_mixed"], flip=fq["flip"], common=fq["common"],
                       d_pp=fq["++"]["delta"], d_pm=fq["+-"]["delta"], d_mp=fq["-+"]["delta"], d_mm=fq["--"]["delta"])
            bs = []
            for _ in range(n_boot):
                gb = np.median(l2[rng.integers(0, l2.shape[0], l2.shape[0])], axis=0)
                q = fit.fit_four_quadrants(ang, gb)
                bs.append((q["flip"], q["common"], q["p_same"], q["p_mixed"]))
            bs = np.array(bs)
            row.update(flip_se=float(np.nanstd(bs[:, 0])), common_se=float(np.nanstd(bs[:, 1])))
        else:
            row.update(p_paper=fit.fit_quadrant(ang, g)["p"])
            bs = [fit.fit_quadrant(ang, np.median(l2[rng.integers(0, l2.shape[0], l2.shape[0])], axis=0))["p"]
                  for _ in range(n_boot)]
            row.update(p_se=float(np.nanstd(bs)))
        rows.append(row)
    summ = {}
    for fa in sorted({r["family"] for r in rows}) + ["ALL"]:
        rs = [r for r in rows if fa == "ALL" or r["family"] == fa]
        s = dict(n=len(rs), p_paper=med_ci([r["p_paper"] for r in rs]))
        if signed:
            for k in ("p_same", "p_mixed", "flip", "common"):
                s[k] = med_ci([r[k] for r in rs])
            s["frac_flip_negative"] = float(np.nanmean([r["flip"] < 0 for r in rs]))
            s["frac_common_negative"] = float(np.nanmean([r["common"] < 0 for r in rs]))
        summ[fa] = s
    return dict(kind="grid", meta={k: meta[k] for k in ("family", "n_anchors", "L", "M", "path")}, summary=summ, rows=rows)


def analyze_kway(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    ang = np.array(meta["angles_deg"], float)
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "k*_set*.npz"))):
        z = np.load(f)
        k = len(z["names"])
        thr, alpha_i = float(z["thr"]), z["alpha_i"]
        if "mix_normed" not in z.files or not np.isfinite(alpha_i).all():
            continue
        s = np.sin(np.deg2rad(alpha_i))
        t = fit.intercept(ang, np.median(z["mix_normed"], axis=0), thr)
        r = np.sqrt(k) * np.sin(np.deg2rad(t)) / np.linalg.norm(s)
        te = fit.intercept(ang, np.median(z["mix_equal"], axis=0), thr)
        xe = np.sin(np.deg2rad(te)) / np.sqrt(k) / s                 # normalised coordinates of the equal mix
        out.setdefault(k, []).append(dict(norm_radius=float(r), equal_l2=float(np.linalg.norm(xe))))
    summ = {}
    for k, v in sorted(out.items()):
        r = med_ci([x["norm_radius"] for x in v])
        summ[k] = dict(n_sets=len(v), norm_radius=r, p_implied=[float(fit.kway_exponent(k, x)) for x in r])
    return dict(kind="kway", meta=dict(family=meta["family"]), summary=summ)


def main():
    for d in sys.argv[1:]:
        meta = json.load(open(os.path.join(d, "meta.json")))
        res = analyze_kway(d) if "ks" in meta else analyze_grid(d)
        json.dump(res, open(os.path.join(d, "analysis.json"), "w"), indent=1)
        print(f"\n== {d}  ({res['kind']}, {res['meta']})")
        if res["kind"] == "grid":
            for fa, s in res["summary"].items():
                line = f"  {fa:10s} n={s['n']:4d}  p(paper quadrant) {s['p_paper'][0]:.3f} [{s['p_paper'][1]:.3f}, {s['p_paper'][2]:.3f}]"
                if "flip" in s:
                    line += (f"  p same-sign {s['p_same'][0]:.3f}  p mixed-sign {s['p_mixed'][0]:.3f}"
                             f"  flip {s['flip'][0]:+.3f} [{s['flip'][1]:+.3f}, {s['flip'][2]:+.3f}]"
                             f"  common {s['common'][0]:+.3f} [{s['common'][1]:+.3f}, {s['common'][2]:+.3f}]")
                print(line)
        else:
            for k, s in res["summary"].items():
                print(f"  k={k:2d} sets={s['n_sets']:3d}  normalised radius {s['norm_radius'][0]:.3f} "
                      f"[{s['norm_radius'][1]:.3f}, {s['norm_radius'][2]:.3f}]  implied p {s['p_implied'][0]:.2f}")


if __name__ == "__main__":
    main()
