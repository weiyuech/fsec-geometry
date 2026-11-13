# Reproducing the FSEC audit

Run commands from the repository root. The published [result tables](results/README.md) and ready-made figures are available in the checkout. Rendering the figures uses those tables. Refitting the contours requires the separate response-grid release. Generating new measurements additionally requires model snapshots, source texts and a GPU.

## Redraw the four figures on a CPU

The plotting script reads the published tables in `results/` and a single measured grid in `examples/`. From an environment with Python 3.11 or newer:

```sh
python -m pip install numpy scipy matplotlib contourpy
python scripts/make_figures.py
```

The script writes PNGs to `figures/`. For vector figures, use `python scripts/make_figures.py --format pdf --out work/print`.

## Explore the same measured grid

Open [the French × tense explorer](https://weiyuech.github.io/fsec-geometry/) or open `docs/index.html` locally in a browser. The page includes the plotted contour and the first crossing at each integer angle. To rebuild its data and the README animation from `examples/Tense__French.npz`, run:

```sh
python scripts/export_explorer.py
python scripts/make_quadrant_preview.py
```

The horizontal axis shows French and the vertical axis shows tense. The source array stores those axes in the opposite order. The export uses the same per-sign axis scaling and the same threshold across the four quadrants as the static figure. The diagonal distance is interpolated from the measured contour. The exponent is a separate fit across angles in a quadrant.

## Install and check the implementation

Use Python 3.11. The experiment environment specified PyTorch 2.8.0 and Transformers 4.57.6. The CPU checks also passed on PyTorch 2.2.2, Transformers 4.57.6, NumPy 1.26.4, SciPy 1.17.1 and contourpy 1.3.3. For the GPU run, install a PyTorch build compatible with your hardware before installing the remaining dependencies.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python tests/test_cpu.py
python run_queue.py jobs/main.json --dry
```

The 44 offline checks exercise known synthetic contours, the cached and full forward paths of four tiny randomly initialized transformer architectures, and a short toy denoiser sweep. To run a job end to end without downloading anything:

```sh
python run_queue.py jobs/smoke.json --only smoke_toy --device cpu --results raw/offline_smoke
```

## Refit the published measurements

Download `fsec-response-grids-v1.tar.gz` from the repository release and place it in the repository root. The archive contains 24 main-queue jobs, 2,478 numeric NPZ files and 240 toy cells. Each cell contains summaries from the simulation, including finite-fit counts and mixture radii. The original GPU verifier checked the main and smoke queues against their job specifications. The public release contains the main queue.

```sh
shasum -a 256 -c results/raw_release.sha256
tar -xzf fsec-response-grids-v1.tar.gz
python reanalysis/gpu_shape.py
python reanalysis/language_sign_check.py
python reanalysis/kway.py > reanalysis/kway_gpu.json
python reanalysis/gpu_reliability.py --cache raw/fit_cache
python reanalysis/h4_controls.py
python reanalysis/toy_sweep.py --require-complete > reanalysis/toy_gpu.json
```

The CPU fits write to `reanalysis/`. The signed threshold and anchor-bootstrap calculations can take several minutes or longer. `gpu_shape.py` fits the paper's positive-positive quadrant at its own threshold and all four quadrants at a shared threshold. `language_sign_check.py` resamples anchor indices jointly across pairs. `kway.py` uses the lower median for both the axes and mixtures, matching the GPU calculation used to select each set's threshold. The original exploratory `analyze.py` uses the midpoint median for mixture curves. `gpu_reliability.py` fits positive-quadrant median grids from the 300-anchor job. `h4_controls.py` compares the same ordered 90 pairs across 18 conditions.

The archive includes per-anchor responses and per-job digests of ordered token sequences. It records their sequence length and unique count. For comparisons of refitted direction vectors, the archive provides the measured same-label alignment scalars. Running new measurements creates the original token-ID arrays and direction-vector files locally. H4 then calculates alignment from those vectors directly. See the [per-file SHA-256 manifest](results/raw_manifest.json).

## Re-run the GPU measurements

The authors' prompt registry, lists of contrastive prompts and published sweep files come from commit `a9355d4001500708a40a255bacfe559e6dc269c0`. Obtain that checkout separately, keeping the upstream license with it.

```sh
mkdir -p external
git clone https://github.com/FranciscoHS/fsec-paper.git external/fsec-paper
git -C external/fsec-paper checkout --detach a9355d4001500708a40a255bacfe559e6dc269c0
git -C external/fsec-paper rev-parse HEAD
```

Set `FSEC_PAPER` to the checkout's path if it lives elsewhere. To reproduce the historical analysis of the authors' archived grids (including their 318-pair median of 2.301), run:

```sh
python -I reanalysis/reliability.py external/fsec-paper reanalysis/reliability.json
```

The audit used 3,000 FineWeb-Edu examples from 30 recorded windows of the `sample-10BT` training split. The fetcher reconstructs the original ordered JSONL with the first 600 characters of each document. It checks the full SHA-256 before publishing the file. Five-token anchor prefixes and random-difference pools use separate seeded draws from this file. The original dataset revision was not archived. A digest mismatch indicates that the current public rows do not reproduce the recorded input. To use an existing copy, set `FSEC_ANCHORS` to its path and verify it explicitly. [Dataset source](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

```sh
python scripts/prepare_anchors.py
python scripts/prepare_anchors.py --verify-only --output raw/anchors.jsonl
```

The full queue uses Gemma-2-2B/9B, GPT-2/large, Qwen3-1.7B and five Pythia sizes at `main` and `step0`. Instructions for accessing the gated Gemma checkpoints are on the model cards. Use your usual local Hugging Face login for those downloads. Preparing just the smoke manifest fetches its three checkpoints. The main manifest fetches all 15 snapshots. The download script records the commits it resolves in `raw/model_revisions.json` and reuses them on subsequent runs. The historical job settings preserve requested revision names. The original weights' immutable commit IDs were unavailable for this release, so fresh Hub downloads can differ from the weights used in the audit.

```sh
python scripts/prepare_models.py jobs/smoke.json
python run_queue.py jobs/smoke.json --results raw/results_smoke
python reanalysis/verify_gpu.py --jobs jobs/smoke.json --results raw/results_smoke
python scripts/prepare_models.py jobs/main.json
python run_queue.py jobs/main.json --results raw/fresh_main
python reanalysis/verify_gpu.py --jobs jobs/main.json --results raw/fresh_main
```

The model loader uses the local Hub cache offline. Set `FSEC_MODELS` if the resolved revision manifest has a different location. Jobs use GPU bfloat16 by default. Each job records the perturbation and measurement layers, angle grid, anchor count and selected execution path. The cached path falls back to a full forward when its numerical check exceeds tolerance. The historical 24-job queue took about 6 h 15 min on one RTX PRO 6000 Blackwell Server Edition GPU, including a resumed toy sweep.

Separate smoke and main directories are important because both specifications contain a job named `smoke_gemma2b` with different pair sets. Fresh measurements should also live separately from extracted release arrays. Rerun the same manifest with the same output directory to resume a stopped experiment. Completed jobs carry a `DONE` marker and completed pair grids or toy cells are reused. Use a fresh directory for changed model or experiment settings. The runner logs failed jobs and exits nonzero if any selected job fails.
