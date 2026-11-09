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
| D-06 | 2026-11-09 | Score all 1003 images; annotate a seeded random 100 plus case candidates, blind to heatmaps and scores | Exceeds the 50-image minimum; MIT1003 has no labels, so correctness needs human judgment. Factor statistics use the random sample only | Superseded by D-21 | A |
| D-07 | 2026-11-09 | Occlusion probe on selected cases to support any shortcut claim | A heatmap alone can't show reliance on a region (spec §6) | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-08 | 2026-11-09 | Rollout: mean head fusion, no discard ratio | Closest to the original formulation; avoids another tunable | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-09 | 2026-11-09 | Same process and stack as HW3: uv + Python 3.11, Scrum-lite, `AGENTS.md` canonical, @ALLEN07056 reviews PRs, macOS + WSL | Worked for HW3; requested by @hwupu | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-10 | 2026-11-09 | Docs live in `docs/` (HW3 used `doc/`) | Confirmed by @hwupu; matches the folder already created | Accepted | A |
| D-11 | 2026-11-09 | Final numbers come from one full run on one machine; the other only cross-checks | Avoids mixing device differences in one table (as HW3 D-12) | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-12 | 2026-11-09 | Commit `scores.csv`, `predictions.csv`, `annotations/`; gitignore `data/` and `results/maps/` | B can analyse without recomputing; maps are large and regenerable | Accepted (B-02; sign-off relayed by A, 2026-11-09) | A |
| D-13 | 2026-11-09 | Human maps: Gaussian σ = 24 native px on the fixation points from `fixPts` (cut at 127) | Dataset docs give no blur size or viewing geometry. Fit against its own `fixMap` on 100 images: best at 24 (mean CC 0.99985). All 1003 images ≥ 0.99. Points are single pixels, unchanged for cuts of 64/127/200 (40 images) | Accepted (A, on PR #1; B has not reviewed) | A |
| D-14 | 2026-11-09 | GitHub repo is public | Requested by @hwupu; same as HW3. Dataset and overlay figures stay out of git | Accepted | A |
| D-15 | 2026-11-09 | Add a `grid` reference per explainer (human map passed through its grid); also compare ResNet and ViT with ViT reduced to 7×7 | Found in B-04: over all 1003 human maps, passing through 7×7 caps mean CC at 0.786 and 14×14 at 0.941, so a direct comparison is confounded by grid size | Accepted (A, on PR #2; B has not reviewed) | A |
| D-16 | 2026-11-09 | Weights: `resnet50.tv2_in1k`, `vit_base_patch16_224.augreg2_in21k_ft_in1k`, both at 224×224 | Modern ImageNet-1k weights with a 7×7 and 14×14 grid at 224. Measured on 100 images, no-crop vs timm's default crop: top-1 agrees 66% (ResNet) and 55% (ViT), mean confidence nearly equal | Accepted (A, on PR #3; B has not reviewed) | A |
| D-17 | 2026-11-09 | Grad-CAM: target logit (pre-softmax), `grad-cam` package up to the ReLU, no min-max scaling, stored on the raw grid | The package's normal output scales each map, turning an all-zero map into a valid-looking one and losing the grid. Checked against an independent implementation | Accepted (A, on PR #4; B has not reviewed) | A |
| D-18 | 2026-11-09 | Keep center-crop sensitivity (B-25) instead of cutting it early | No-crop changes the top-1 class on many images (D-16), so the choice needs measuring | Accepted | A |
| D-19 | 2026-11-09 | sAUC negatives: canvas fixations of all other images; ties count as half | No random sampling, so scores are deterministic; a center prior then scores about 0.5 (tested) | Accepted (A, on PR #5; B has not reviewed) | A |
| D-20 | 2026-11-09 | Blind labelling: 139 images (random 100 with seed 2026, plus 45 candidates, 6 overlap) in shuffled order; annotators see only the image and predicted classes, as "Model A/B". The shortlist stays out of git until labelling is done | Hides which images were chosen for low scores. Random-sample statistics stay independent of scores | Superseded by D-21 | A |
| D-21 | 2026-11-09 | Drop hand labelling. Still score all 1003 images. Factors: confidence within each model, ResNet-50 and ViT agreement, human fixation concentration. Faces, text and correctness are discussed only for the final cases | @hwupu tried it and found it too costly. HW4 requires considering such factors, not labelling. Cost: no measured result for content or correctness across images, stated as a limitation | Accepted (A; B has not reviewed) | A |
| D-22 | 2026-11-09 | Rank explainers by category. Categories use image and human data only: face (YuNet, score ≥ 0.75), text (EasyOCR, area ≥ 0.5%), small salient attention, edge/texture-rich, peripheral attention (20% tails). Effects are inside minus outside on the within-explainer CC + NSS percentile. Candidates come only from Holm-significant effects | Proposed by @hwupu. Map-derived categories would be circular. Edge and texture correlate at 0.88, so they are one category. Contextual shortcuts and task-irrelevant regions cannot be ranked and are left to case analysis | Accepted (A, on PR #12; B has not reviewed) | A |
| D-23 | 2026-11-10 | Occlusion on all 1003 images, not only the cases. Masks: top 20% of the human map or explainer map, against 5 random blobs of equal area. Fills: heavy blur and mean color. Target: the model's top-1 class | Forward passes are cheap, and a dataset result gives the cases context. Two fills show whether a result depends on the replacement | Proposed | A |
| D-24 | 2026-11-10 | Final cases: the five top-gap images in `annotations/case_notes.csv` | Accepted by @hwupu from the D-22 shortlist; B has not reviewed | Accepted (A) | A |

## Template

```
| D-xx | YYYY-MM-DD | <what we chose> | <why; alternatives considered> | Proposed | <A/B> |
```
