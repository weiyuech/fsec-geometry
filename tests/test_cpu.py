"""CPU checks that need no downloaded weights. Run:  python tests/test_cpu.py

1. fit: recovers known exponents; four-quadrant statistics separate a superellipse
   from a quadratic form with a cross term; k-way exponent law.
2. engine: on tiny randomly initialised GPT-2 / GPT-NeoX (Pythia) / Gemma-2 / Qwen3
   models, the cached-prefix path equals the full path, equals an independent
   block-by-block forward (GPT-2), zero perturbation gives zero response, and the
   rotation preserves the norm and has the requested angle.
3. toy: runs end to end on a small configuration.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fsa import fit
from fsa.engine import Engine, tangent_basis, exp_map, sweep

torch.manual_seed(0)
OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, detail, flush=True)


# ---- 1. fits -----------------------------------------------------------------
def synth_grid(ang, fn):
    # same chart as the engine: rotate by r = |(a1, a2)| toward the direction (a1, a2) / r,
    # so the tangential displacement is sin(r) * (a1, a2) / r
    a = np.deg2rad(ang)
    A1, A2 = np.meshgrid(a, a, indexing="ij")
    r = np.hypot(A1, A2)
    sc = np.where(r > 1e-12, np.sin(r) / np.where(r > 1e-12, r, 1.0), 1.0)
    return fn(sc * A1, sc * A2)


def test_fit():
    pos = np.arange(0, 61, 2.0)
    sgn = np.concatenate([-pos[:0:-1], pos])
    for p in (2.0, 2.3, 3.0, 4.0):
        g = synth_grid(pos, lambda x, y: (np.abs(1.3 * x) ** p + np.abs(0.8 * y) ** p) ** (1 / p))
        f = fit.fit_quadrant(pos, g)
        check(f"superellipse p={p} recovered", abs(f["p"] - p) < 0.03 * p, f"got {f['p']:.3f}")
    g = synth_grid(sgn, lambda x, y: (np.abs(x) ** 2.4 + np.abs(y) ** 2.4) ** (1 / 2.4))
    f = fit.fit_four_quadrants(sgn, g)
    check("superellipse: no flip between quadrants", abs(f["flip"]) < 0.01, f"flip={f['flip']:.4f} common={f['common']:.4f}")
    check("superellipse: p equal in same/mixed quadrants", abs(f["p_same"] - f["p_mixed"]) < 0.03,
          f"{f['p_same']:.3f} vs {f['p_mixed']:.3f}")
    g = synth_grid(sgn, lambda x, y: np.sqrt(x * x + y * y - 0.16 * x * y))       # delta = -0.08
    f = fit.fit_four_quadrants(sgn, g)
    check("tilted ellipse: one quadrant looks like p>2", 2.2 < f["++"]["p"] < 2.5, f"p(++)={f['++']['p']:.3f}")
    check("tilted ellipse: flip detected", abs(f["flip"] + 0.16) < 0.02, f"flip={f['flip']:.4f} (expect -0.16)")
    check("tilted ellipse: mixed quadrants look like p<2", f["p_mixed"] < 1.9, f"p_mixed={f['p_mixed']:.3f}")
    for p in (2.3, 3.0, 6.0):
        for k in (2, 8, 16):
            r = k ** (0.5 - 1 / p)
            check(f"kway exponent p={p} k={k}", abs(float(fit.kway_exponent(k, r)) - p) < 1e-6)


# ---- 2. engine ---------------------------------------------------------------
def tiny_models():
    from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM,
                              Gemma2Config, Gemma2ForCausalLM, Qwen3Config, Qwen3ForCausalLM)
    yield "gpt2", GPT2LMHeadModel(GPT2Config(n_layer=6, n_embd=64, n_head=4, vocab_size=200, n_positions=32))
    yield "gpt_neox", GPTNeoXForCausalLM(GPTNeoXConfig(num_hidden_layers=6, hidden_size=64, num_attention_heads=4,
                                                         intermediate_size=128, vocab_size=200, max_position_embeddings=32))
    yield "gemma2", Gemma2ForCausalLM(Gemma2Config(num_hidden_layers=6, hidden_size=64, num_attention_heads=4,
                                                    num_key_value_heads=2, head_dim=16, intermediate_size=128,
                                                    vocab_size=200, max_position_embeddings=32, sliding_window=16))
    yield "qwen3", Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=6, hidden_size=64, num_attention_heads=4,
                                                 num_key_value_heads=2, head_dim=16, intermediate_size=128,
                                                 vocab_size=200, max_position_embeddings=32))


def test_engine():
    for name, model in tiny_models():
        model = model.float().eval()
        ids = torch.randint(3, 200, (6, 5))
        eng = Engine(model, perturb_layer=1, batch=64, mode="auto", log=lambda *a: None)
        eng.set_anchors(ids)
        check(f"{name}: cached-prefix path active", eng._path == "kv")
        D = eng.anchors.shape[1]
        dirs = torch.randn(3, D)
        basis = tangent_basis(eng.anchors, dirs)
        gram = torch.einsum("nid,njd->nij", basis, basis)
        check(f"{name}: tangent basis orthonormal and tangent",
              (gram - torch.eye(3)).abs().max() < 1e-4 and
              torch.einsum("nkd,nd->nk", basis, eng.anchors).abs().max() < 1e-3)
        coords = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, -0.4, 0.1], [-0.5, 0.5, 0.5]])
        pts = exp_map(eng.anchors, basis, coords)
        R = eng.anchors.norm(dim=-1)
        check(f"{name}: rotation preserves norm", ((pts.norm(dim=-1) - R[:, None]).abs() / R[:, None]).max() < 1e-5)
        cosang = (pts * eng.anchors[:, None, :]).sum(-1) / (R[:, None] ** 2)
        check(f"{name}: rotation angle equals |coords|",
              (torch.arccos(cosang.clamp(-1, 1)) - coords.norm(dim=-1)[None, :]).abs().max() < 2e-3)
        r_kv = sweep(eng, basis, coords)
        check(f"{name}: zero perturbation -> zero response", r_kv[:, 0].abs().max() < 1e-4, f"{r_kv[:, 0].abs().max():.2e}")
        eng._path = "full"
        eng.ref = eng._respond(torch.arange(6), eng.anchors)
        r_full = sweep(eng, basis, coords)
        check(f"{name}: cached-prefix == full path", (r_kv - r_full).abs().max() < 1e-3 * r_full.max(),
              f"max diff {(r_kv - r_full).abs().max():.2e}")
        if name == "gpt2":                                   # independent block-by-block forward
            with torch.no_grad():
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
                h = hs[eng.L + 1].clone()
                ref = hs[eng.M + 1][:, -1, :]
                manual = []
                for g in range(coords.shape[0]):
                    hh = h.clone()
                    hh[:, -1, :] = pts[:, g, :]
                    for blk in model.transformer.h[eng.L + 1: eng.M + 1]:
                        hh = blk(hh)[0]
                    manual.append((hh[:, -1, :] - ref).norm(dim=-1))
                manual = torch.stack(manual, dim=1)
            check("gpt2: hook path == manual block-by-block forward",
                  (manual - r_full).abs().max() < 1e-3 * r_full.max(), f"max diff {(manual - r_full).abs().max():.2e}")
        eng.close()


# ---- 3. toy ------------------------------------------------------------------
def test_toy():
    import tempfile, json
    from fsa.toy import run_toy
    with tempfile.TemporaryDirectory() as d:
        n = run_toy(d, ds=(4096,), anchor_noises=(0.0, 0.1732), amaxes=(25.0,), thetas_deg=(0,), ks=(2, 4),
                    n_anchors=4, n_sets=6, n_steps=21, seeds=(0,), device="cpu", log=lambda *a: print("   ", *a))
        rows = json.load(open(os.path.join(d, "summary.json")))
    pc = [r["p_median"] for r in rows if r["kind"] == "pair" and r["anchor_noise"] == 0][0]
    pn = [r["p_median"] for r in rows if r["kind"] == "pair" and r["anchor_noise"] > 0][0]
    check("toy: paper regime (clean anchors, magnitudes to 25) gives p well above 2", n > 0 and pc > 2.3, f"p={pc:.2f}")
    print(f"    toy with noisy anchors: p={pn:.2f}")
    for r in rows:
        if r["kind"] == "kway":
            print(f"    kway anchor_noise={r['anchor_noise']} k={r['k']}: radius {r['norm_radius']:.3f} -> p_implied {r['p_implied']:.2f}")


if __name__ == "__main__":
    test_fit()
    test_engine()
    test_toy()
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    sys.exit(0 if all(OK) else 1)
