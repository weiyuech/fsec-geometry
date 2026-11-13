"""Models, anchors and direction families."""
import json
import os
import sys
import hashlib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.environ.get("FSEC_PAPER", os.path.join(ROOT, "external", "fsec-paper"))

# Short model label -> (Hugging Face repository, requested revision).
MODELS = {
    "gemma9b": ("google/gemma-2-9b", "main"),
    "gemma2b": ("google/gemma-2-2b", "main"),
    "qwen": ("Qwen/Qwen3-1.7B", "main"),
    "gpt2": ("openai-community/gpt2", "main"),
    "gpt2-large": ("openai-community/gpt2-large", "main"),
}
for _s in ("70m", "160m", "410m", "1b", "1.4b"):
    MODELS[f"pythia-{_s}"] = (f"EleutherAI/pythia-{_s}", "main")
    MODELS[f"pythia-{_s}-step0"] = (f"EleutherAI/pythia-{_s}", "step0")


def snapshot_path(name):
    from huggingface_hub import snapshot_download

    repo, revision = MODELS[name]
    manifest = os.environ.get("FSEC_MODELS", os.path.join(ROOT, "raw", "model_revisions.json"))
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as handle:
            record = json.load(handle)[name]
        if record["repo"] != repo or record["revision"] != revision:
            raise ValueError(f"Model manifest entry differs from the job's requested revision for {name}")
        revision = record["sha"]
    return snapshot_download(repo_id=repo, revision=revision, local_files_only=True)


def load_model(name, device="cuda", dtype="bfloat16"):
    """name: a key of MODELS, or a local directory. Loads offline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = name if os.path.isdir(name) else snapshot_path(name)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype), local_files_only=True)
    except TypeError:                                    # older transformers
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, dtype),
                                                     local_files_only=True)
    return model.to(device).eval(), tok


def shuffle_weights(model, seed):
    """Authors' randomized-transformer control: permute entries within every 2D block matrix."""
    from .engine import find_blocks
    g = torch.Generator(device="cpu").manual_seed(seed)
    named = sorted(((f"block{i}.{n}", p) for i, b in enumerate(find_blocks(model))
                    for n, p in b.named_parameters() if p.ndim == 2), key=lambda t: t[0])
    with torch.no_grad():
        for _, p in named:
            flat = p.detach().reshape(-1)
            perm = torch.randperm(flat.numel(), generator=g).to(flat.device)
            p.copy_(flat[perm].reshape(p.shape))
    return len(named)


# ---- anchors -----------------------------------------------------------------
def load_texts(path=None):
    path = path or os.environ.get("FSEC_ANCHORS") or os.path.join(ROOT, "raw", "anchors.jsonl")
    return [json.loads(l)["text"] for l in open(path, encoding="utf-8")]


def anchor_ids(tokenizer, texts, n, seq_len=5, seed=0, skip=0):
    """First `seq_len` tokens (special tokens included, as in the paper) of n shuffled texts.

    `skip` drops that many usable texts first, so disjoint anchor sets can be drawn
    from one seed. Returns LongTensor [n, seq_len].
    """
    order = np.random.default_rng(seed).permutation(len(texts))
    out, skipped = [], 0
    for i in order:
        ids = tokenizer(texts[i], truncation=True, max_length=seq_len, add_special_tokens=True)["input_ids"]
        if len(ids) != seq_len:
            continue
        if skipped < skip:
            skipped += 1
            continue
        out.append(ids)
        if len(out) == n:
            break
    if len(out) < n:
        raise ValueError(f"only {len(out)} usable anchors, wanted {n}")
    return torch.tensor(out, dtype=torch.long)


# ---- directions --------------------------------------------------------------
def _registry():
    if not os.path.isfile(os.path.join(UPSTREAM, "scripts", "lib", "registry.py")):
        raise FileNotFoundError("Set FSEC_PAPER to the checkout in REPRODUCING.md")
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    from scripts.lib import registry
    return registry


def contrastive_names(exclude=("Formal", "HonestyShort", "TensePresent")):
    return [n for n in _registry().all_direction_names() if n not in exclude]


def contrastive_directions(engine, tokenizer, names=None):
    """Difference-of-means directions (b - a) from the paper's exact prompt pairs. Unit norm."""
    reg = _registry()
    names = names or contrastive_names()
    out, meta = {}, {}
    for n in names:
        a, b = reg.get_prompts(n)
        m = min(len(a), len(b))
        a, b = list(a[:m]), list(b[:m])
        d = (engine.last_token_acts(tokenizer, b) - engine.last_token_acts(tokenizer, a)).mean(0)
        out[n] = d / d.norm()
        meta[n] = dict(n_pairs=m, sha=hashlib.sha256("\x00".join(a + b).encode()).hexdigest()[:16])
    return out, meta


def randdiff_directions(pool_acts, n_dirs, k_avg=30, seed=0):
    """Construction-matched control: mean of k_avg differences of randomly paired activations."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for i in range(n_dirs):
        ia = torch.randint(0, pool_acts.shape[0], (k_avg,), generator=g)
        ib = torch.randint(0, pool_acts.shape[0], (k_avg,), generator=g)
        d = (pool_acts[ia.to(pool_acts.device)] - pool_acts[ib.to(pool_acts.device)]).mean(0)
        out[f"rdiff_{i:03d}"] = d / d.norm()
    return out


def random_directions(dim, n_dirs, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(n_dirs, dim, generator=g)
    v = v / v.norm(dim=-1, keepdim=True)
    return {f"rand_{i:03d}": v[i].to(device) for i in range(n_dirs)}


def low_overlap_pairs(dirs, max_cos=0.1, limit=None, seed=0, min_pairs=None):
    """Pairs with |cos| < max_cos. If fewer than min_pairs qualify, the least-overlapping
    pairs are used instead (their cos is saved with each result, so analysis can filter)."""
    names = sorted(dirs)
    V = torch.stack([dirs[n] for n in names])
    C = (V @ V.T).abs().cpu().numpy()
    allp = [(C[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    pairs = [(a, b) for c, a, b in allp if c < max_cos]
    want = min_pairs if min_pairs is not None else (limit or 1)
    want = min(want, len(allp))
    if len(pairs) < want:
        pairs = [(a, b) for c, a, b in sorted(allp)[:want]]
    if limit is not None and len(pairs) > limit:
        idx = np.random.default_rng(seed).choice(len(pairs), limit, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    return pairs


def low_overlap_sets(dirs, k, n_sets, max_cos=0.1, seed=0, tries=3000):
    """Random sets of k directions with all pairwise |cos| below max_cos; the bound is
    relaxed stepwise if no set exists. Returns (sets, bound_used)."""
    names = sorted(dirs)
    if k > len(names):
        return [], None
    V = torch.stack([dirs[n] for n in names])
    C = (V @ V.T).abs().cpu().numpy()
    rng = np.random.default_rng(seed)
    for bound in [b for b in (max_cos, 0.2, 0.3, 0.5, 1.01) if b >= max_cos]:
        found = {}
        for _ in range(tries):
            chosen = []
            for i in rng.permutation(len(names)):
                if all(C[i, j] < bound for j in chosen):
                    chosen.append(int(i))
                    if len(chosen) == k:
                        break
            if len(chosen) == k:
                found[tuple(sorted(chosen))] = [names[i] for i in chosen]
                if len(found) == n_sets:
                    break
        if found:
            return list(found.values()), bound
    return [], None
