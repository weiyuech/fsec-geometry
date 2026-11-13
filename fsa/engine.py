"""Batched activation-perturbation engine for HF causal LMs.

For N anchor prompts of equal token length T we replace the last-token residual
stream after block `perturb_layer` with arbitrary vectors and read the last-token
residual after block `measure_layer` (same convention as fsec-paper:
hidden_states[layer + 1]).

Two execution paths, checked against each other at start-up:
  kv   : the T-1 prefix tokens are run once and cached; each perturbed point is a
         single-token forward on top of the cached prefix (about T times cheaper).
  full : every point re-runs the whole T-token sequence (what the paper's code does).

Everything stays on the GPU; batches are sized to fill it and halve on OOM.
"""
import copy
import torch


class _Stop(Exception):
    pass


def find_blocks(model):
    for path in ("transformer.h", "gpt_neox.layers", "model.layers", "model.decoder.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"cannot locate transformer blocks in {type(model).__name__}")


class Engine:
    def __init__(self, model, perturb_layer=2, measure_layer=None, batch=4096, mode="auto",
                 log=print):
        self.model = model.eval()
        self.blocks = find_blocks(model)
        self.L = perturb_layer
        self.M = len(self.blocks) - 2 if measure_layer is None else measure_layer
        if self.M < 0:
            self.M += len(self.blocks)
        assert 0 <= self.L < self.M < len(self.blocks), (self.L, self.M, len(self.blocks))
        self.dev = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.batch = batch
        self.mode = mode
        self.log = log
        self._inj = None
        self._meas = None
        self._hooks = [self.blocks[self.L].register_forward_hook(self._inject_hook),
                       self.blocks[self.M].register_forward_hook(self._measure_hook)]
        self.n_forward_points = 0

    def close(self):
        for h in self._hooks:
            h.remove()

    # ---- hooks -------------------------------------------------------------
    def _inject_hook(self, mod, args, out):
        if self._inj is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = self._inj
        return out

    def _measure_hook(self, mod, args, out):
        if self._inj is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        self._meas = h[:, -1, :].float()
        raise _Stop

    # ---- anchors -----------------------------------------------------------
    @torch.inference_mode()
    def set_anchors(self, input_ids):
        """input_ids: LongTensor [N, T]. Caches prefix state, anchor activations, references."""
        self.ids = input_ids.to(self.dev)
        N, T = self.ids.shape
        self.T = T
        acts, self.cache = [], None
        self._inj = None
        for s in range(0, N, 256):
            out = self.model(input_ids=self.ids[s:s + 256], output_hidden_states=True, use_cache=False)
            acts.append(out.hidden_states[self.L + 1][:, -1, :].float())
        self.anchors = torch.cat(acts)                      # [N, D] fp32
        if self.mode in ("auto", "kv") and T > 1:
            try:
                self.cache = self.model(input_ids=self.ids[:, :-1], use_cache=True).past_key_values
                self._path = "kv"
                self._selfcheck()
            except Exception as e:                           # any API or numerical mismatch
                if self.mode == "kv":
                    raise
                self.log(f"[engine] kv path disabled ({type(e).__name__}: {str(e)[:120]}); using full path")
                self.cache, self._path = None, "full"
        else:
            self._path = "full"
        idx = torch.arange(N, device=self.dev)
        self.ref = self._respond(idx, self.anchors)          # [N, Dm] fp32
        self.log(f"[engine] anchors N={N} T={T} path={self._path} L={self.L} M={self.M} "
                 f"dtype={self.dtype} |a| mean={self.anchors.norm(dim=-1).mean():.3f}")
        return self.anchors

    def _selfcheck(self):
        n = min(self.ids.shape[0], 8)
        idx = torch.arange(n, device=self.dev)
        g = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.randn(self.anchors[:n].shape, generator=g).to(self.dev)
        pts = self.anchors[:n] + 0.3 * self.anchors[:n].norm(dim=-1, keepdim=True) * noise / noise.norm(dim=-1, keepdim=True)
        a = self._forward_kv(idx, pts)
        b = self._forward_full(idx, pts)
        rel = ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-9)).max().item()
        tol = 1e-3 if self.dtype == torch.float32 else 3e-2
        if not rel < tol:
            raise RuntimeError(f"kv/full mismatch rel={rel:.3e} tol={tol}")
        self.log(f"[engine] kv path verified against full path (max rel diff {rel:.2e})")

    # ---- forwards ----------------------------------------------------------
    def _run(self, **kw):
        try:
            self.model(**kw)
        except _Stop:
            pass
        out, self._meas, self._inj = self._meas, None, None
        return out

    def _forward_full(self, idx, pts):
        self._inj = pts.to(self.dtype)
        return self._run(input_ids=self.ids[idx], use_cache=False)

    def _forward_kv(self, idx, pts):
        cache = copy.deepcopy(self.cache)
        cache.batch_select_indices(idx)
        self._inj = pts.to(self.dtype)
        mask = torch.ones((idx.shape[0], self.T), dtype=torch.long, device=self.dev)
        return self._run(input_ids=self.ids[idx, -1:], past_key_values=cache,
                         attention_mask=mask, use_cache=True)

    @torch.inference_mode()
    def _respond(self, idx, pts):
        fwd = self._forward_kv if self._path == "kv" else self._forward_full
        outs, s = [], 0
        while s < idx.shape[0]:
            b = self.batch
            try:
                outs.append(fwd(idx[s:s + b], pts[s:s + b]))
                s += b
            except torch.cuda.OutOfMemoryError:
                self._inj = self._meas = None
                torch.cuda.empty_cache()
                if self.batch <= 16:
                    raise
                self.batch //= 2
                self.log(f"[engine] OOM -> batch {self.batch}")
        self.n_forward_points += idx.shape[0]
        return torch.cat(outs)

    @torch.inference_mode()
    def response_l2(self, idx, pts):
        """L2 distance at the measure layer between perturbed and unperturbed runs. [P] fp32."""
        return (self._respond(idx, pts) - self.ref[idx]).norm(dim=-1)

    @torch.inference_mode()
    def last_token_acts(self, tokenizer, prompts):
        """Clean last-token residual after block L for prompts of any length (batch 1, no padding)."""
        self._inj = None
        out = []
        for p in prompts:
            tok = tokenizer(p, return_tensors="pt").to(self.dev)
            hs = self.model(**tok, output_hidden_states=True, use_cache=False).hidden_states
            out.append(hs[self.L + 1][0, -1, :].float())
        return torch.stack(out)


# ---- geometry (all on GPU, fp32) -------------------------------------------
def tangent_basis(anchors, dirs):
    """Gram-Schmidt the k directions in the tangent plane of each anchor.

    anchors [N, D]; dirs [k, D] (same for all anchors) or [N, k, D]. Order matters:
    direction 0 is kept, direction 1 is orthogonalised against it, and so on
    (identical to fsec-paper's orthonormalize for k = 2). Returns [N, k, D].
    """
    a_hat = anchors / anchors.norm(dim=-1, keepdim=True)
    if dirs.dim() == 2:
        dirs = dirs.unsqueeze(0).expand(anchors.shape[0], -1, -1)
    basis = []
    for i in range(dirs.shape[1]):
        v = dirs[:, i, :] - (dirs[:, i, :] * a_hat).sum(-1, keepdim=True) * a_hat
        for e in basis:
            v = v - (v * e).sum(-1, keepdim=True) * e
        n = v.norm(dim=-1, keepdim=True)
        if (n < 1e-6 * dirs[:, i, :].norm(dim=-1, keepdim=True)).any():
            raise ValueError(f"direction {i} is degenerate in the tangent plane of some anchor")
        basis.append(v / n)
    return torch.stack(basis, dim=1)


def exp_map(anchors, basis, coords):
    """Norm-preserving rotation. coords [G, k] in radians -> points [N, G, D].

    a' = R (cos r a_hat + sin r v / r),  v = coords @ basis,  r = |coords|.
    """
    R = anchors.norm(dim=-1, keepdim=True)                       # [N, 1]
    a_hat = anchors / R
    r = coords.norm(dim=-1)                                      # [G]
    scale = torch.where(r > 1e-12, torch.sin(r) / r.clamp_min(1e-12), torch.ones_like(r))
    v = torch.einsum("gk,nkd->ngd", coords * scale[:, None], basis)
    return R[:, None, :] * (torch.cos(r)[None, :, None] * a_hat[:, None, :] + v)


@torch.inference_mode()
def sweep(engine, basis, coords, anchor_chunk=None):
    """Response for every (anchor, coordinate). basis [N, k, D], coords [G, k] rad -> [N, G] (GPU)."""
    N, G = basis.shape[0], coords.shape[0]
    D = basis.shape[-1]
    if anchor_chunk is None:                                     # keep the point buffer under ~6 GB
        anchor_chunk = max(1, int(6e9 // (G * D * 4)))
    out = []
    for s in range(0, N, anchor_chunk):
        sl = slice(s, min(s + anchor_chunk, N))
        pts = exp_map(engine.anchors[sl], basis[sl], coords)     # [n, G, D]
        n = pts.shape[0]
        idx = torch.arange(sl.start, sl.stop, device=pts.device).repeat_interleave(G)
        out.append(engine.response_l2(idx, pts.reshape(n * G, D)).reshape(n, G))
        del pts
    return torch.cat(out)
