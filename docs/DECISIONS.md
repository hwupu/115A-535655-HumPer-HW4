# Decision Log

Lightweight ADRs. Newest at the bottom. Status: `Proposed` → `Accepted` / `Rejected` / `Superseded by D-xx`.
A `Proposed` decision needs both members to agree at the next planning/review. Details for D-01..D-08 are in `ARCHITECTURE.md` §4, §5 and §8.

| ID | Date | Decision | Why | Status | Owner |
|---|---|---|---|---|---|
| D-01 | 2026-11-09 | Dataset: MIT1003 (fallback: CAT2000 train) | Real eye-tracking on natural photos; small download. SALICON's attention comes from mouse tracking. CAT2000 has many non-ImageNet categories | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-02 | 2026-11-09 | ResNet-50 Grad-CAM, ViT-B/16 Grad-CAM, ViT-B/16 Attention Rollout, explaining the top-1 prediction | Overlapping pairs separate architecture from method; no ground-truth labels exist | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-03 | 2026-11-09 | Whole-image resize to 224×224 (no center crop); compare on a 224×224 canvas; no thresholding | Center crop discards real fixations and biases alignment. Center-crop subset is a sensitivity check | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-04 | 2026-11-09 | Metrics: CC + NSS main, sAUC supplementary; every score reported against `center` and `shuffled` baselines | Map-based + fixation-based; sAUC and baselines control for center bias | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-05 | 2026-11-09 | Human maps: Gaussian blur σ ≈ 1° of visual angle, built at native resolution | Standard practice. Exact pixels to be confirmed from dataset documentation in B-03 | Superseded by D-13 | A |
| D-06 | 2026-11-09 | Score all 1003 images; annotate a seeded random 100 plus case candidates, blind to heatmaps and scores | Exceeds the 50-image minimum; MIT1003 has no labels, so correctness needs human judgment. Factor statistics use the random sample only | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-07 | 2026-11-09 | Occlusion probe on selected cases to support any shortcut claim | A heatmap alone can't show reliance on a region (spec §6) | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-08 | 2026-11-09 | Rollout: mean head fusion, no discard ratio | Closest to the original formulation; avoids another tunable | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-09 | 2026-11-09 | Same process and stack as HW3: uv + Python 3.11, Scrum-lite, `AGENTS.md` canonical, @ALLEN07056 reviews PRs, macOS + WSL | Worked for HW3; requested by @hwupu | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-10 | 2026-11-09 | Docs live in `docs/` (HW3 used `doc/`) | Confirmed by @hwupu; matches the folder already created | Accepted | A |
| D-11 | 2026-11-09 | Final numbers come from one full run on one machine; the other only cross-checks | Avoids mixing device differences in one table (as HW3 D-12) | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-12 | 2026-11-09 | Commit `scores.csv`, `predictions.csv`, `annotations/`; gitignore `data/` and `results/maps/` | B can analyse without recomputing; maps are large and regenerable | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-13 | 2026-11-09 | Human maps: Gaussian σ = 24 native px on the fixation points from `fixPts` (cut at 127) | Dataset docs give no blur size or viewing geometry. Fit against its own `fixMap` on 100 images: best at 24 (mean CC 0.99985). All 1003 images ≥ 0.99. Points are single pixels, unchanged for cuts of 64/127/200 (40 images) | Proposed | A |
| D-14 | 2026-11-09 | GitHub repo is public | Requested by @hwupu; same as HW3. Dataset and overlay figures stay out of git | Accepted | A |

## Template

```
| D-xx | YYYY-MM-DD | <what we chose> | <why; alternatives considered> | Proposed | <A/B> |
```
