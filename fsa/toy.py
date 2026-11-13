"""Toy model of error correction with known features (GPU, self-contained).

Same model as the paper's toy (fsec-paper/toy_model, Vaintrob's random-sparse denoiser):

    f(x) = c_out * E^T tanh^3(c_in * E x),   E in {0, +1, -1}^{H x d}, P(nonzero) = q, tied decoder

built to denoise x = two-hot + Gaussian noise (variance 0.03).
"paper" presets use the (q, c_in, c_out) the authors selected for H = 1024.

The paper measures the boundary on ONE noise-free anchor with perturbation
magnitudes up to 25. Here the same measurement is swept over
  anchor noise     0 (paper) ... 0.173 (the noise level the model was built to correct)
  magnitude range  1.5 ... 25 (paper)
  rotation theta   of the probe directions away from the true features
and extended to signed grids (four quadrants) and k-feature mixtures.
"""
import json
import os
import numpy as np
import torch

from .fit import fit_four_quadrants, intercept, kway_exponent

# d -> (q, c_in, c_out), from fsec-paper toy_model/runs/misalignment_sweep_H1024/phase2_d*_H1024.npz
PAPER = {4096: (0.15, 0.24875, 0.034443), 8192: (0.10, 0.24875, 0.039417), 16384: (0.075, 0.1825, 0.046067)}
ACT = {"tanh3": lambda z: torch.tanh(z) ** 3, "relu": torch.relu, "relu2": lambda z: torch.relu(z) ** 2}


class Toy:
    def __init__(self, d, H=1024, q=None, c_in=None, c_out=None, act="tanh3", seed=0, device="cuda",
                 train_noise=0.03 ** 0.5):
        if q is None:
            q, c_in, c_out = PAPER[d]
        g = torch.Generator(device="cpu").manual_seed(seed)
        u = torch.rand(H, d, generator=g)
        E = torch.zeros(H, d)
        E[u < q / 2] = 1.0
        E[(u >= q / 2) & (u < q)] = -1.0
        self.E = E.to(device)
        self.d, self.H, self.q, self.dev, self.act = d, H, q, device, ACT[act]
        self.c_in, self.c_out, self.train_noise = c_in, c_out, train_noise
        self.g = torch.Generator(device=device).manual_seed(seed + 1)

    def sample(self, n):
        x = torch.zeros(n, self.d, device=self.dev)
        idx = torch.rand(n, self.d, device=self.dev, generator=self.g).topk(2, dim=1).indices
        x.scatter_(1, idx, 1.0)
        return x, idx

    def randn(self, shape):
        return torch.randn(shape, device=self.dev, generator=self.g)

    def f(self, x):
        return self.c_out * (self.act(self.c_in * (x @ self.E.T)) @ self.E)

    @torch.inference_mode()
    def quality(self, n=1024):
        """How well the model denoises at its training noise: active output, inactive |output|."""
        x, _ = self.sample(n)
        y = self.f(x + self.train_noise * self.randn(x.shape))
        on = x > 0.5
        return dict(on_mean=float(y[on].mean()), off_abs_mean=float(y[~on].abs().mean()),
                    input_off_abs_mean=float(self.train_noise * (2 / np.pi) ** 0.5))

    @torch.inference_mode()
    def response(self, x0, V, coords, max_rows=1 << 15):
        """x0 [N, d] anchors, V [k, d] probe directions, coords [G, k] -> L2 response [N, G]."""
        ref = self.f(x0)
        P = coords @ V
        chunk = max(1, max_rows // P.shape[0])
        out = []
        for s in range(0, x0.shape[0], chunk):
            xs = x0[s:s + chunk]
            y = self.f((xs[:, None, :] + P[None, :, :]).reshape(-1, self.d))
            out.append((y.reshape(xs.shape[0], P.shape[0], self.d) - ref[s:s + chunk, None, :]).norm(dim=-1))
        return torch.cat(out)


def _inactive_sets(idx, d, k, n_sets, rng):
    active = set(idx.flatten().tolist())
    free = np.array([i for i in range(d) if i not in active])
    return [rng.choice(free, k, replace=False) for _ in range(n_sets)]


def _probe(toy, feats, theta, rng):
    """Orthonormal probe directions rotated by theta (rad) from feature axes toward random ones."""
    k = len(feats)
    V = torch.zeros(k, toy.d, device=toy.dev)
    V[torch.arange(k), torch.as_tensor(np.asarray(feats), device=toy.dev)] = 1.0
    if theta > 0:
        W = torch.tensor(rng.standard_normal((k, toy.d)), dtype=torch.float32, device=toy.dev)
        V = np.cos(theta) * V + np.sin(theta) * W / W.norm(dim=-1, keepdim=True)
        V = torch.linalg.qr(V.T).Q.T
    return V


def measure(toy, anchor_noise, amax, theta_deg, ks, n_anchors, n_sets, n_steps, rng):
    dev = toy.dev
    x, idx = toy.sample(n_anchors)
    x0 = x + anchor_noise * toy.randn(x.shape)
    mag = np.linspace(0, amax, n_steps)
    smag = np.concatenate([-mag[:0:-1], mag])
    mt = torch.tensor(mag, dtype=torch.float32, device=dev)
    st = torch.tensor(smag, dtype=torch.float32, device=dev)
    S1, S2 = torch.meshgrid(st, st, indexing="ij")
    grid = torch.stack([S1.flatten(), S2.flatten()], -1)
    th = np.deg2rad(theta_deg)
    out = []
    ps, pm, flips, commons = [], [], [], []
    for feats in _inactive_sets(idx, toy.d, 2, n_sets, rng):
        g = toy.response(x0, _probe(toy, feats, th, rng), grid).median(0).values
        fq = fit_four_quadrants(smag, g.reshape(len(smag), len(smag)).cpu().numpy(), linear=True)
        ps.append(fq["++"]["p"]); pm.append(fq["p_mixed"]); flips.append(fq["flip"]); commons.append(fq["common"])
    if np.isfinite(ps).any():
        out.append(dict(kind="pair", p_median=float(np.nanmedian(ps)),
                        p_q25=float(np.nanpercentile(ps, 25)), p_q75=float(np.nanpercentile(ps, 75)),
                        p_mixed_median=float(np.nanmedian(pm)), flip_median=float(np.nanmedian(flips)),
                        common_median=float(np.nanmedian(commons)), n_fit=int(np.isfinite(ps).sum())))
    for k in ks:
        radii = []
        for feats in _inactive_sets(idx, toy.d, k, max(6, n_sets // 3), rng):
            V = _probe(toy, feats, th, rng)
            eye = torch.eye(k, device=dev)
            ax = toy.response(x0, V, (mt[None, :, None] * eye[:, None, :]).reshape(-1, k))
            med = ax.median(0).values.reshape(k, len(mag)).cpu().numpy()
            thr = 0.5 * med.max(1).min()
            s = np.array([intercept(mag, med[i], thr) for i in range(k)])
            if not np.isfinite(s).all():
                continue
            w = torch.tensor(s / np.linalg.norm(s), dtype=torch.float32, device=dev)
            t = intercept(mag, toy.response(x0, V, mt[:, None] * w[None, :]).median(0).values.cpu().numpy(), thr)
            radii.append(np.sqrt(k) * t / np.linalg.norm(s))     # ||x||_2 in per-axis normalised units
        if radii and np.isfinite(radii).any():
            r = float(np.nanmedian(radii))
            out.append(dict(kind="kway", k=k, norm_radius=r, p_implied=float(kway_exponent(k, r))))
    return out


def run_toy(out_dir, ds=(4096, 8192, 16384), H=1024, anchor_noises=(0.0, 0.0433, 0.0866, 0.1732),
            amaxes=(1.5, 3.0, 8.0, 25.0), thetas_deg=(0, 15, 30, 45, 60, 90), ks=(2, 3, 4, 6, 8, 12, 16, 24, 32),
            n_anchors=128, n_sets=60, n_steps=41, seeds=(0, 1, 2, 3, 4), acts=("tanh3",), device="cuda",
            log=print, **_):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for act in acts:
        for d in ds:
            for seed in seeds:
                toy = Toy(d, H, act=act, seed=seed, device=device)
                qual = toy.quality()
                for an in anchor_noises:
                    for amax in amaxes:
                        tag = f"{act}_d{d}_s{seed}_an{an}_amax{amax}"
                        fout = os.path.join(out_dir, tag + ".json")
                        if os.path.exists(fout):
                            rows.extend(json.load(open(fout)))
                            continue
                        rng = np.random.default_rng(seed)
                        cur = []
                        for th in thetas_deg:
                            for r in measure(toy, an, amax, th, ks, n_anchors, n_sets, n_steps, rng):
                                cur.append(dict(r, act=act, d=d, H=H, seed=seed, anchor_noise=an, amax=amax,
                                                theta=th, **qual))
                        json.dump(cur, open(fout, "w"))
                        rows.extend(cur)
                        p0 = [round(c["p_median"], 2) for c in cur if c["kind"] == "pair" and c["theta"] == 0]
                        log(f"[toy] {tag}: p(theta=0)={p0}  on={qual['on_mean']:.2f} off|.|={qual['off_abs_mean']:.3f}")
    json.dump(rows, open(os.path.join(out_dir, "summary.json"), "w"))
    return len(rows)
