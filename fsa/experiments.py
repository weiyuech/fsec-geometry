"""GPU experiments. Each function is resumable: finished items are skipped.

grid     signed (four-quadrant) or positive 2D iso-response grids for direction pairs
kway     k-direction mixtures: axis sweeps, then equal and sensitivity-normalised mixes
"""
import json
import os
import time
import numpy as np
import torch

from . import data
from .engine import Engine, tangent_basis, sweep


def _save(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _angles(max_deg, step_deg, signed):
    n = int(round(max_deg / step_deg))
    pos = np.arange(0, n + 1) * step_deg
    return np.concatenate([-pos[:0:-1], pos]) if signed else pos


def build_directions(engine, tok, family, texts, n_dirs=40, seed=0, pool_n=1500, seq_len=5):
    """Returns ({name: unit vector on GPU}, meta)."""
    if family == "contrastive":
        return data.contrastive_directions(engine, tok)
    if family == "randdiff":
        ids = data.anchor_ids(tok, texts, pool_n, seq_len=seq_len, seed=1000 + seed)
        acts = []
        with torch.inference_mode():
            for s in range(0, ids.shape[0], 256):
                hs = engine.model(input_ids=ids[s:s + 256].to(engine.dev), output_hidden_states=True,
                                  use_cache=False).hidden_states
                acts.append(hs[engine.L + 1][:, -1, :].float())
        return data.randdiff_directions(torch.cat(acts), n_dirs, seed=seed), dict(pool_n=pool_n)
    if family == "random":
        D = engine.model.get_input_embeddings().weight.shape[1]
        return data.random_directions(D, n_dirs, seed=seed, device=engine.dev), {}
    raise ValueError(family)


def run_grid(engine, tok, out_dir, family="contrastive", pairs=None, n_anchors=30, anchor_seed=0,
             anchor_skip=0, signed=True, max_deg=60.0, step_deg=2.0, max_cos=0.1, limit=None,
             n_dirs=40, dir_seed=0, texts_path=None, log=print, **_):
    os.makedirs(out_dir, exist_ok=True)
    texts = data.load_texts(texts_path)
    engine.set_anchors(data.anchor_ids(tok, texts, n_anchors, seed=anchor_seed, skip=anchor_skip))
    dirs, dmeta = build_directions(engine, tok, family, texts, n_dirs=n_dirs, seed=dir_seed)
    if pairs is None:
        pairs = data.low_overlap_pairs(dirs, max_cos=max_cos, limit=limit, seed=dir_seed)
    else:
        pairs = [tuple(p) for p in pairs if p[0] in dirs and p[1] in dirs]
    if not pairs:
        raise RuntimeError(f"no direction pairs to run for family {family}")
    log(f"[grid:{family}] {len(pairs)} pairs, {len(dirs)} directions, {n_anchors} anchors")
    ang = _angles(max_deg, step_deg, signed)
    a = torch.tensor(np.deg2rad(ang), dtype=torch.float32, device=engine.dev)
    A1, A2 = torch.meshgrid(a, a, indexing="ij")
    coords = torch.stack([A1.flatten(), A2.flatten()], dim=-1)
    torch.save({k: v.cpu() for k, v in dirs.items()}, os.path.join(out_dir, "directions.pt"))
    json.dump(dict(family=family, pairs=pairs, n_anchors=n_anchors, anchor_seed=anchor_seed,
                   anchor_skip=anchor_skip, angles_deg=ang.tolist(), L=engine.L, M=engine.M,
                   path=engine._path, dir_meta=dmeta, anchor_ids=engine.ids.cpu().tolist()),
              open(os.path.join(out_dir, "meta.json"), "w"))
    t0, done = time.time(), 0
    for i, (n1, n2) in enumerate(pairs):
        f = os.path.join(out_dir, f"{n1}__{n2}.npz")
        if os.path.exists(f):
            continue
        basis = tangent_basis(engine.anchors, torch.stack([dirs[n1], dirs[n2]]))
        l2 = sweep(engine, basis, coords).reshape(n_anchors, len(ang), len(ang))
        _save(f, l2=l2.cpu().numpy().astype(np.float32), angles_deg=ang,
              cos12=float((dirs[n1] @ dirs[n2]).item()))
        done += 1
        if done == 1 or done % 10 == 0:
            el = time.time() - t0
            rate = done * coords.shape[0] * n_anchors / el
            log(f"[grid:{family}] {i + 1}/{len(pairs)} pairs  {rate:,.0f} pts/s  "
                f"eta {(len(pairs) - i - 1) * el / done / 60:.1f} min  batch={engine.batch}")
    return len(pairs)


def run_kway(engine, tok, out_dir, family="contrastive", ks=(2, 3, 4, 6, 8, 12, 16), n_sets=24,
             n_anchors=30, anchor_seed=0, max_deg=88.0, step_deg=1.0, frac=0.5, max_cos=0.1,
             n_dirs=40, dir_seed=0, texts_path=None, log=print, **_):
    """For each random low-overlap set of k directions:

      stage 1  sweep every direction alone  -> boundary angle alpha_i at the set's threshold
      stage 2  sweep two mixtures:
                 equal   all tangent coordinates equal
                 normed  coordinates proportional to sin(alpha_i), i.e. every direction
                         contributes the same fraction of its own boundary
    Saved curves are medians over anchors plus the per-anchor arrays.
    """
    os.makedirs(out_dir, exist_ok=True)
    texts = data.load_texts(texts_path)
    engine.set_anchors(data.anchor_ids(tok, texts, n_anchors, seed=anchor_seed))
    dirs, _m = build_directions(engine, tok, family, texts, n_dirs=n_dirs, seed=dir_seed)
    ang = _angles(max_deg, step_deg, False)
    a = torch.tensor(np.deg2rad(ang), dtype=torch.float32, device=engine.dev)
    json.dump(dict(family=family, ks=list(ks), n_sets=n_sets, n_anchors=n_anchors, angles_deg=ang.tolist(),
                   frac=frac, L=engine.L, M=engine.M), open(os.path.join(out_dir, "meta.json"), "w"))
    from .fit import intercept
    n_done = 0
    for k in ks:
        sets, bound = data.low_overlap_sets(dirs, k, n_sets, max_cos=max_cos, seed=dir_seed + k)
        if not sets:
            log(f"[kway:{family}] cannot form sets of size {k} from {len(dirs)} directions; skipping")
            continue
        if bound > max_cos:
            log(f"[kway:{family}] k={k}: overlap bound relaxed to {bound}")
        n_done += len(sets)
        for si, names in enumerate(sets):
            f = os.path.join(out_dir, f"k{k:02d}_set{si:03d}.npz")
            if os.path.exists(f):
                continue
            basis = tangent_basis(engine.anchors, torch.stack([dirs[n] for n in names]))
            eye = torch.eye(k, device=engine.dev)
            axis_coords = (a[None, :, None] * eye[:, None, :]).reshape(k * len(ang), k)
            axis = sweep(engine, basis, axis_coords).reshape(n_anchors, k, len(ang))
            med = axis.median(dim=0).values.cpu().numpy()                    # [k, A]
            thr = frac * float(med.max(axis=1).min())
            alpha_i = np.array([intercept(ang, med[i], thr) for i in range(k)])
            s = np.sin(np.deg2rad(alpha_i))
            mixes = {"equal": np.ones(k) / np.sqrt(k)}
            if np.isfinite(s).all():
                mixes["normed"] = s / np.linalg.norm(s)
            res = dict(axis=axis.cpu().numpy().astype(np.float32), alpha_i=alpha_i, thr=thr, overlap_bound=bound,
                       names=np.array(names), angles_deg=ang)
            for label, m in mixes.items():
                mc = a[:, None] * torch.tensor(m, dtype=torch.float32, device=engine.dev)[None, :]
                curve = sweep(engine, basis, mc)                              # [N, A]
                res[f"mix_{label}"] = curve.cpu().numpy().astype(np.float32)
                res[f"w_{label}"] = m
            _save(f, **res)
        log(f"[kway:{family}] k={k}: {len(sets)} sets done")
    if n_done == 0:
        raise RuntimeError(f"no direction sets could be formed for family {family}")
    return True


EXPERIMENTS = {"grid": run_grid, "kway": run_kway}


def make_engine(model_name, perturb_layer=2, measure_layer=None, batch=4096, mode="auto",
                shuffle_seed=None, device="cuda", dtype="bfloat16", log=print):
    model, tok = data.load_model(model_name, device=device, dtype=dtype)
    if shuffle_seed is not None:
        n = data.shuffle_weights(model, shuffle_seed)
        log(f"[model] shuffled {n} block matrices (seed {shuffle_seed})")
    return Engine(model, perturb_layer, measure_layer, batch=batch, mode=mode, log=log), tok
