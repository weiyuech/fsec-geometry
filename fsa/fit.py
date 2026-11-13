"""Contour extraction and shape fits. numpy/scipy/contourpy only (CPU).

Conventions follow FranciscoHS/fsec-paper (scripts/lib/superellipse.py):
grid[i, j] = response at (alpha_1 = ang[i], alpha_2 = ang[j]); per-pair
threshold = frac * min(axis maxima); exact-geodesic normalisation.
"""
import numpy as np
import contourpy
from scipy.optimize import minimize_scalar


def intercept(ang, curve, thr):
    """First angle (linear interpolation) where a 1D curve starting at ang[0] crosses thr."""
    if not np.isfinite(curve).all() or curve.max() < thr:
        return np.nan
    for i in range(1, len(curve)):
        if (curve[i - 1] - thr) * (curve[i] - thr) <= 0:
            y0, y1 = curve[i - 1], curve[i]
            if y1 == y0:
                return float(ang[i - 1])
            return float(ang[i - 1] + (thr - y0) / (y1 - y0) * (ang[i] - ang[i - 1]))
    return np.nan


def quadrant(ang, grid, s1=1, s2=1):
    """Cut one quadrant out of a signed grid and return it in (|alpha|, |alpha|) form.

    ang must be symmetric about 0 and contain 0. s1, s2 in {+1, -1}.
    """
    i0 = int(np.argmin(np.abs(ang)))
    assert abs(ang[i0]) < 1e-9, "signed grid must contain 0"
    ii = np.arange(i0, len(ang)) if s1 > 0 else np.arange(i0, -1, -1)
    jj = np.arange(i0, len(ang)) if s2 > 0 else np.arange(i0, -1, -1)
    return np.abs(ang[ii]), grid[np.ix_(ii, jj)]


def norm_points(ang, grid, thr, linear=False):
    """Iso-response contour of a positive-quadrant grid in per-axis normalised coordinates.

    linear=False: ang is a rotation angle in degrees (exact-geodesic sin normalisation,
    1 degree edge margin), as in the paper. linear=True: ang is an additive magnitude.
    """
    lines = contourpy.contour_generator(ang, ang, grid.T).lines(thr)
    if not lines:
        return None
    raw = np.concatenate(lines, axis=0)
    gm = float(ang.max())
    edge = gm / 60.0 if linear else 1.0
    raw = raw[(raw[:, 0] < gm - edge) & (raw[:, 1] < gm - edge)]
    t1, t2 = intercept(ang, grid[:, 0], thr), intercept(ang, grid[0, :], thr)
    if len(raw) < 3 or not (np.isfinite(t1) and np.isfinite(t2)) or t1 <= 0 or t2 <= 0:
        return None
    if linear:
        x, y = raw[:, 0] / t1, raw[:, 1] / t2
    else:
        r = np.hypot(raw[:, 0], raw[:, 1])
        rs = np.where(r > 1e-12, r, 1.0)
        x = np.sin(np.deg2rad(r)) / np.sin(np.deg2rad(t1)) * raw[:, 0] / rs
        y = np.sin(np.deg2rad(r)) / np.sin(np.deg2rad(t2)) * raw[:, 1] / rs
    m = (x > 0.05) & (y > 0.05) & (x < 1.5) & (y < 1.5)
    if m.sum() < 3:
        return None
    return x[m], y[m], t1, t2


def pair_threshold(ang, grid, frac=0.5):
    """Authors' per-pair threshold on a positive-quadrant grid."""
    return frac * min(grid[:, 0].max(), grid[0, :].max())


def fit_superellipse(x, y, bounds=(0.5, 10.0)):
    r = minimize_scalar(lambda p: float(np.sum((x ** p + y ** p - 1.0) ** 2)),
                        bounds=bounds, method="bounded")
    p = float(r.x)
    resid = float(np.abs((x ** p + y ** p) ** (1.0 / p) - 1.0).mean())
    return p, resid


def fit_tilted_ellipse(x, y):
    """x^2 + 2 delta x y + y^2 = 1, least squares in delta."""
    a, b = x * x + y * y - 1.0, 2.0 * x * y
    delta = -float(np.sum(a * b) / np.sum(b * b))
    v = np.clip(x * x + 2 * delta * x * y + y * y, 1e-9, None)
    return delta, float(np.abs(np.sqrt(v) - 1.0).mean())


def fit_quadrant(ang, grid, thr=None, frac=0.5, linear=False):
    """p, delta and residuals for one positive-quadrant grid (median over anchors already taken)."""
    if thr is None:
        thr = pair_threshold(ang, grid, frac)
    pts = norm_points(ang, grid, thr, linear)
    if pts is None:
        return dict(p=np.nan, resid_p=np.nan, delta=np.nan, resid_delta=np.nan,
                    t1=np.nan, t2=np.nan, thr=float(thr), n=0)
    x, y, t1, t2 = pts
    p, rp = fit_superellipse(x, y)
    d, rd = fit_tilted_ellipse(x, y)
    return dict(p=p, resid_p=rp, delta=d, resid_delta=rd, t1=t1, t2=t2, thr=float(thr), n=int(len(x)))


def fit_four_quadrants(ang, grid, frac=0.5, linear=False):
    """Fit each quadrant of a signed grid with ONE shared threshold.

    The threshold is frac * the smallest of the four half-axis maxima, so every
    half-axis reaches it. Returns per-quadrant fits plus the two statistics that
    separate the hypotheses:

      superellipse (L^p in the two directions)  ->  p equal in all four quadrants,
                                                    delta has the same sign in all four
      quadratic form with a cross term          ->  delta flips sign between
                                                    same-sign (++, --) and mixed (+-, -+) quadrants

    flip = mean delta(++,--) - mean delta(+-,-+)   (0 under superellipse)
    common = mean of the four deltas               (0 under pure quadratic form)
    """
    quads = {"++": (1, 1), "+-": (1, -1), "-+": (-1, 1), "--": (-1, -1)}
    cut = {k: quadrant(ang, grid, *s) for k, s in quads.items()}
    thr = frac * min(min(g[:, 0].max(), g[0, :].max()) for _, g in cut.values())
    out = {k: fit_quadrant(a, g, thr=thr, linear=linear) for k, (a, g) in cut.items()}
    d = {k: out[k]["delta"] for k in quads}
    out["flip"] = float(np.mean([d["++"], d["--"]]) - np.mean([d["+-"], d["-+"]]))
    out["common"] = float(np.mean(list(d.values())))
    out["p_same"] = float(np.mean([out["++"]["p"], out["--"]["p"]]))
    out["p_mixed"] = float(np.mean([out["+-"]["p"], out["-+"]["p"]]))
    out["thr"] = float(thr)
    return out


def kway_exponent(k, ratio):
    """Exponent implied by a k-way sensitivity-normalised mixture.

    Under R = F(sum_i w_i |c_i|^p), mixing k directions so that every normalised
    coordinate c_i / s_i is equal gives  ||x||_2 = k^(1/2 - 1/p)  at the boundary,
    where s_i is direction i's own boundary. ratio = measured ||x||_2.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 0.5 - np.log(ratio) / np.log(k)
        return np.where(inv > 1e-6, 1.0 / inv, np.inf)
