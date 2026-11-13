"""CPU-only reanalysis of the authors' cached sweeps (FranciscoHS/fsec-paper).

Question: how reliable is the superellipse exponent p as a *measurement*?
  A. replication of the headline medians (pipeline check)
  B. split-half reliability over the 30 anchors (same pairs, disjoint anchors)
  C. bootstrap SE per pair vs between-pair spread
  D. alternate-forms: does a pair's p carry over to other anchor sources,
     layers, metrics and models?
  E. per-anchor fits: on how many individual anchors is p > 2?
  F. is p a property of directions?  additive model p_ij = mu + a_i + a_j

Pickles are read with a restricted unpickler (numpy + plain containers only).
Run:  python -I reliability.py /path/to/fsec-paper  [out.json]
"""
import sys, os, glob, json, pickle, csv, itertools
import numpy as np
import contourpy
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

REPO = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "reliability.json"
SW = os.path.join(REPO, "results", "sweeps_2d")
RNG = np.random.default_rng(0)
EXCLUDE = {"Formal", "HonestyShort", "TensePresent"}


class Restricted(pickle.Unpickler):
    OK = {("builtins", n) for n in
          ("dict", "list", "tuple", "set", "float", "int", "str", "slice", "complex")}

    def find_class(self, module, name):
        if module.split(".")[0] == "numpy" or (module, name) in self.OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load(path):
    with open(path, "rb") as f:
        return Restricted(f).load()


def intercept(ang, curve, thr):
    if curve.max() < thr:
        return np.nan
    for i in range(1, len(curve)):
        if (curve[i - 1] - thr) * (curve[i] - thr) <= 0:
            y0, y1 = curve[i - 1], curve[i]
            if y1 == y0:
                return float(ang[i - 1])
            return float(ang[i - 1] + (thr - y0) / (y1 - y0) * (ang[i] - ang[i - 1]))
    return np.nan


def norm_points(ang, grid, frac=0.5):
    """Canonical protocol: per-pair threshold, exact-geodesic normalisation."""
    thr = frac * min(grid[:, 0].max(), grid[0, :].max())
    cg = contourpy.contour_generator(ang, ang, grid.T)
    lines = cg.lines(thr)
    if not lines:
        return None
    raw = np.concatenate(lines, axis=0)
    gm = float(ang.max())
    raw = raw[(raw[:, 0] < gm - 1.0) & (raw[:, 1] < gm - 1.0)]
    if len(raw) < 3:
        return None
    t1, t2 = intercept(ang, grid[:, 0], thr), intercept(ang, grid[0, :], thr)
    if not (np.isfinite(t1) and np.isfinite(t2)) or t1 <= 0 or t2 <= 0:
        return None
    r = np.hypot(raw[:, 0], raw[:, 1])
    rs = np.where(r > 1e-12, r, 1.0)
    x = np.sin(np.deg2rad(r)) / np.sin(np.deg2rad(t1)) * raw[:, 0] / rs
    y = np.sin(np.deg2rad(r)) / np.sin(np.deg2rad(t2)) * raw[:, 1] / rs
    m = (x > 0.05) & (y > 0.05) & (x < 1.5) & (y < 1.5)
    return (x[m], y[m]) if m.sum() >= 3 else None


def fit_p(ang, grid, frac=0.5):
    pts = norm_points(ang, grid, frac)
    if pts is None:
        return np.nan
    x, y = pts
    r = minimize_scalar(lambda p: float(np.sum((x ** p + y ** p - 1) ** 2)),
                        bounds=(0.5, 10.0), method="bounded")
    return float(r.x)


def fit_delta(ang, grid, frac=0.5):
    pts = norm_points(ang, grid, frac)
    if pts is None:
        return np.nan
    x, y = pts
    a, b = x * x + y * y - 1.0, 2 * x * y
    return -float(np.sum(a * b) / np.sum(b * b))


def pairs_of(target, layer, suffix=""):
    pre, suf = f"sweep2d_{target}_L{layer}_", f"_fineweb_60deg{suffix}.pkl"
    out = {}
    for p in glob.glob(os.path.join(SW, pre + "*" + suf)):
        stem = os.path.basename(p)[len(pre):-len(suf)]
        if "__" not in stem or "_fineweb_" in stem:
            continue
        a, b = stem.split("__", 1)
        out[(a, b)] = p
    return out


def keep_set():
    k = set()
    for row in csv.DictReader(open(os.path.join(REPO, "results", "pos_pairs_gemma_L2.csv"))):
        if row["a"] in EXCLUDE or row["b"] in EXCLUDE:
            continue
        k.add((row["a"], row["b"])); k.add((row["b"], row["a"]))
    return k


def med(g):
    return np.median(g, axis=0)


def main():
    res = {}
    keep = keep_set()
    canon = {k: v for k, v in pairs_of("gemma", 2).items() if k in keep}
    print("canonical contrastive pairs:", len(canon), flush=True)
    data = {k: load(v) for k, v in canon.items()}
    ang = next(iter(data.values()))["angles_deg"].astype(float)
    keys = sorted(data)

    # A. replication
    p_full = np.array([fit_p(ang, med(data[k]["l2"])) for k in keys])
    d_full = np.array([fit_delta(ang, med(data[k]["l2"])) for k in keys])
    ok = np.isfinite(p_full)
    res["A_replication"] = dict(n=int(ok.sum()), median_p=float(np.median(p_full[ok])),
                                mean_p=float(p_full[ok].mean()),
                                frac_p_le_2=float((p_full[ok] <= 2).mean()),
                                median_delta=float(np.nanmedian(d_full)),
                                frac_delta_neg=float((d_full[ok] < 0).mean()),
                                spearman_p_vs_delta=float(spearmanr(p_full[ok], d_full[ok])[0]))
    print("A", res["A_replication"], flush=True)

    # B. split-half over anchors
    n_anch = data[keys[0]]["l2"].shape[0]
    rs = []
    for _ in range(60):
        perm = RNG.permutation(n_anch)
        h1, h2 = perm[: n_anch // 2], perm[n_anch // 2:]
        a = np.array([fit_p(ang, med(data[k]["l2"][h1])) for k in keys])
        b = np.array([fit_p(ang, med(data[k]["l2"][h2])) for k in keys])
        m = np.isfinite(a) & np.isfinite(b)
        rs.append((spearmanr(a[m], b[m])[0], np.corrcoef(a[m], b[m])[0, 1],
                   np.median(np.abs(a[m] - b[m]))))
    rs = np.array(rs)
    r_s = float(np.median(rs[:, 0]))
    res["B_split_half"] = dict(spearman_half=r_s, pearson_half=float(np.median(rs[:, 1])),
                               spearman_brown_full=float(2 * r_s / (1 + r_s)),
                               median_abs_diff_between_halves=float(np.median(rs[:, 2])),
                               n_splits=len(rs), anchors_per_half=n_anch // 2)
    print("B", res["B_split_half"], flush=True)

    # C. bootstrap SE per pair
    boots = np.full((len(keys), 60), np.nan)
    for j in range(60):
        idx = RNG.integers(0, n_anch, n_anch)
        boots[:, j] = [fit_p(ang, med(data[k]["l2"][idx])) for k in keys]
    se = np.nanstd(boots, axis=1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=1)
    var_between = np.nanvar(p_full)
    res["C_bootstrap"] = dict(median_se=float(np.nanmedian(se)),
                              sd_between_pairs=float(np.sqrt(var_between)),
                              reliability_ratio=float(1 - np.nanmean(se ** 2) / var_between),
                              frac_pairs_ci_excludes_2_above=float(np.mean(lo > 2)),
                              frac_pairs_ci_excludes_2_below=float(np.mean(hi < 2)),
                              frac_pairs_ci_contains_2=float(np.mean((lo <= 2) & (hi >= 2))))
    print("C", res["C_bootstrap"], flush=True)

    # E. per-anchor fits
    pa = np.array([[fit_p(ang, data[k]["l2"][i]) for i in range(n_anch)] for k in keys])
    res["E_per_anchor"] = dict(frac_anchor_fits_finite=float(np.isfinite(pa).mean()),
                               median_over_all=float(np.nanmedian(pa)),
                               frac_anchor_fits_gt2=float(np.nanmean(pa > 2)),
                               median_pair_frac_gt2=float(np.median(np.nanmean(pa > 2, axis=1))),
                               median_within_pair_sd=float(np.nanmedian(np.nanstd(pa, axis=1))),
                               anchor_main_effect_sd=float(np.nanstd(np.nanmedian(pa, axis=0))))
    print("E", res["E_per_anchor"], flush=True)

    # F. additive direction model
    names = sorted({n for k in keys for n in k})
    ix = {n: i for i, n in enumerate(names)}
    kk = [k for k, o in zip(keys, ok) if o]
    X = np.zeros((len(kk), len(names)))
    for r_, (a, b) in enumerate(kk):
        X[r_, ix[a]] = 1; X[r_, ix[b]] = 1
    yv = p_full[ok] - p_full[ok].mean()
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r2 = 1 - np.sum((yv - X @ coef) ** 2) / np.sum(yv ** 2)
    order = np.argsort(coef)
    res["F_additive"] = dict(r2_direction_main_effects=float(r2), n_dirs=len(names),
                             lowest=[(names[i], round(float(coef[i]), 3)) for i in order[:4]],
                             highest=[(names[i], round(float(coef[i]), 3)) for i in order[-4:]])
    print("F", res["F_additive"], flush=True)

    # D. alternate forms
    base = dict(zip(keys, p_full))
    forms = [("anchors: wikipedia-en", "gemma", 2, "_srcwiki_en"), ("anchors: wikipedia-zh", "gemma", 2, "_srcwiki_zh"),
             ("anchors: code", "gemma", 2, "_srccode"), ("additive perturbation", "gemma", 2, "_additive"),
             ("measure layer -7", "gemma", 2, "_M-7"), ("measure layer -12", "gemma", 2, "_M-12"),
             ("perturb layer 5", "gemma", 5, ""), ("perturb layer 10", "gemma", 10, ""),
             ("perturb layer 20", "gemma", 20, ""), ("model qwen", "qwen", 2, ""),
             ("model llama", "llama", 2, ""), ("model mistral", "mistral", 2, ""),
             ("model aya", "aya", 2, ""), ("model yi", "yi", 2, "")]
    D = {}
    for label, tgt, layer, suf in forms:
        other = pairs_of(tgt, layer, suf)
        a, b = [], []
        for k in keys:
            pth = other.get(k) or other.get((k[1], k[0]))
            if pth is None or not np.isfinite(base[k]):
                continue
            g = load(pth)
            p2 = fit_p(g["angles_deg"].astype(float), med(g["l2"]))
            if np.isfinite(p2):
                a.append(base[k]); b.append(p2)
        if len(a) > 10:
            D[label] = dict(n=len(a), spearman=float(spearmanr(a, b)[0]),
                            median_other=float(np.median(b)),
                            frac_same_side_of_2=float(np.mean((np.array(a) > 2) == (np.array(b) > 2))))
            print("D", label, D[label], flush=True)
    res["D_alternate_forms"] = D

    json.dump(res, open(OUT, "w"), indent=1)
    np.savez(OUT.replace(".json", ".npz"), keys=np.array(["__".join(k) for k in keys]),
             p_full=p_full, d_full=d_full, boots=boots, per_anchor=pa)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
