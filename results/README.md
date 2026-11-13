# Result data

The independent GPU run completed on November 12, 2026. It measured 318 ordered contrastive pairs on Gemma-2-9B, using 30 FineWeb-Edu anchors for the four-quadrant sweep. The replication table matches all 318 pairs against fits from the authors’ published response grids at commit `a9355d4001500708a40a255bacfe559e6dc269c0`. Both columns use the positive-positive quadrant and its own per-pair threshold. The independent run sampled different anchors. [Upstream source](https://github.com/FranciscoHS/fsec-paper/tree/a9355d4001500708a40a255bacfe559e6dc269c0)

| File | Observations and interpretation |
| --- | --- |
| [replication_pairs.csv](replication_pairs.csv) | One row per ordered pair. `upstream_p` fits the authors’ cached grid and `audit_p` fits the independent run. The table contains 132 language–semantic, 112 code–semantic and 74 semantic–semantic pairs. |
| [gpu_shape.json](gpu_shape.json) | Four quadrant fits for each of 318 contrastive pairs and 200 random-difference pairs, with per-family summaries. Quadrant signs follow the axis order within each row’s `pair`. The four fits share a threshold. `p_paper` uses the separate positive-positive threshold. |
| [language_sign_check.json](language_sign_check.json) | Sensitivity of the within-pair language-sign difference to the threshold and to 250 shared resamples of the 30 anchor indices across all 132 pairs. The reported 95% interval conditions on those pairs. |
| [kway_gpu.json](kway_gpu.json) | Two direction families, seven mixture sizes and 24 sets at each size, for 336 sets. `radius` uses the lower median on both axes and mixtures. Each triple gives the point estimate and the lower and upper endpoints of its set-bootstrap interval. |
| [gpu_reliability.json](gpu_reliability.json) | Positive-quadrant fits on 90 prespecified pairs sharing 300 anchors. Includes per-pair intervals, 200 shared anchor bootstraps, 20 disjoint split draws at each sample size, and crossed variance components for single-anchor fits. |
| [h4_controls.json](h4_controls.json) | Matched 90-pair panel, three equal family strata. Contains per-pair estimates and aggregate results for 18 checkpoint or shuffle conditions, eight individual trained/control comparisons and the paired three-shuffle mean comparison. `path` records the model forward method (`kv` or `full`). |
| [toy_gpu.json](toy_gpu.json) | Seed-level summaries over all 240 cells in the size, noise, magnitude and seed sweep. `p_median` is first computed across feature pairs and subsequently summarized across seeds. Five-seed minima and maxima describe the observed range. |
| [toy_pairs.csv](toy_pairs.csv) | All 1,440 pair summaries from 240 cells and six rotation angles per cell. Each row contains statistics from 30 pairs, including their median and quartiles. |
| [toy_kway.csv](toy_kway.csv) | All 12,960 mixture summaries from the same 240 cells, six rotations and nine mixture sizes per cell. |
| [reliability.json](reliability.json) and [family_and_psd.json](family_and_psd.json) | Earlier reanalysis of the upstream cached data. Includes the upstream median, split-half reliability, family summaries and the direction-clique calculation. |

The shape summaries’ pair-bootstrap intervals hold the observed anchors fixed and treat pairs as sampling units, although pairs share directions. The language sensitivity and 300-anchor reliability calculations resample the common anchor indices together. In the toy tables `d` is the input dimension, `H` the hidden dimension, `theta` the rotation in degrees, `anchor_noise` the standard deviation of the starting noise per coordinate, and `amax` the maximum perturbation magnitude per grid coordinate. The primary toy fit uses an initially inactive pair of feature directions and a shared four-quadrant threshold. Its `p_median` summarizes positive-positive fits.

## Response-grid archive

The release asset `fsec-response-grids-v1.tar.gz` contains the measured per-anchor response arrays and experiment metadata for the main queue’s language-model experiments, the toy cell summaries, and a copy of the per-file manifest. Extracting it from the repository root populates `raw/results/`. [Complete file manifest](raw_manifest.json) · [Archive checksum](raw_release.sha256)

```sh
shasum -a 256 -c results/raw_release.sha256
tar -xzf fsec-response-grids-v1.tar.gz
```

The archive contains 2,478 NumPy files, including the four grids from the main queue’s preliminary smoke job. Grid files store `l2[anchor, first_angle, second_angle]`, `angles_deg`, and the raw direction cosine `cos12`. Mixture files store axes, mixtures, normalized weights, threshold and direction identities. The toy saved per-cell summaries during execution. Metadata records the ordered prefix checksum, unique count and prefix length for grid jobs. Same-label trained/control direction cosines are provided as scalars in `raw/results/direction_alignment.json`.
