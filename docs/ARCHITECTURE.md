# Architecture

> **Status: plan.** Only the scaffold and `device.py` exist so far. Choices marked **(D-xx)** are tracked in `docs/DECISIONS.md`. References to "HW3 D-xx" mean the HW3 decision log.

Small analysis project, same shape as HW3. Design goal: **compute expensive maps once, score and analyse many times.**

The question it has to answer (spec §4): *is the model using visual evidence that is consistent with human attention?* The grade rewards showing **where, when and why** human and model attention differ, so a raw average score is not enough. Every score needs a baseline that says what it means.

## 1. Spec → design

| Spec requirement | Design | Rubric |
|---|---|---|
| Public human-attention dataset | MIT1003 (D-01) | Setup 15 |
| Pretrained model, ≥ 1 explanation method | 2 models, 3 explainers (D-02) | Understanding 15 |
| ≥ 50 images | All 1003 scored; random 100 hand-annotated (D-06) | Quant 20 |
| Compatible spatial representation, described clearly | One geometry pipeline in `maps.py`, §4 (D-03) | Setup 15 |
| ≥ 1 metric, 2 encouraged | CC + NSS main, sAUC as center-bias control (D-04) | Quant 20 |
| Inspect individual cases | `failures.py`, rule-based candidates | Misalign 20 |
| ≥ 5 low-alignment cases with image, human map, explanation, score | `viz.case_figure()` | Misalign 20 |
| Shortcut / perceptual-failure reasoning | Occlusion probe gives evidence beyond the heatmap (D-07) | Interpretation 20 |
| Effect of content, correctness, salient objects | Blind annotations + factor tables (D-06) | Interpretation 20 |
| "Attribution is not automatically an explanation" | Model-randomization check + cautious wording rule | Understanding 15 |
| Reproduce reported figures | Scripts → `results/`, notebooks only read from it | Setup 15 |
| 3 pp recommended, 4 max | At most 3 figures + 1 table in the main text | Clarity 10 |

## 2. Repo layout

Mirrors HW3. The one difference is `docs/` instead of `doc/`, matching the folder already created here.

```
115A-535655-HumPer-HW4/
├── README.md                 # setup (macOS + WSL) & run commands
├── AGENTS.md                 # AI agent context (canonical)
├── CLAUDE.md                 # `@AGENTS.md` import only
├── .gitattributes            # LF everywhere
├── pyproject.toml / uv.lock  # uv env (Python 3.11, universal lock)
├── src/humper/               # reusable logic; notebooks import from here
│   ├── device.py             # get_device(): copied from HW3 unchanged
│   ├── data.py               # MIT1003 → tidy DataFrame, fixations, human maps
│   ├── models.py             # model registry + per-model preprocessing
│   ├── explain.py            # explainer registry: Grad-CAM, Attention Rollout
│   ├── maps.py               # geometry and normalization (the only place)
│   ├── alignment.py          # metric registry: CC, NSS, sAUC
│   ├── evaluate.py           # baselines, bootstrap CIs, paired tests, factor tables
│   ├── failures.py           # low-alignment mining, reliability filter, diversity caps
│   ├── probes.py             # occlusion probe, model-randomization check
│   └── viz.py                # case figures, distributions
├── scripts/
│   ├── download_data.py      # fetch + checksum MIT1003 into data/
│   ├── compute_maps.py       # model maps + predictions (resumable)
│   ├── compute_scores.py     # maps → scores.csv (seconds to minutes)
│   └── run_probes.py         # occlusion + randomization on selected images
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_alignment.ipynb       # distributions, baselines
│   ├── 03_failure_cases.ipynb   # candidates
│   ├── 04_selected_cases.ipynb  # final cases + probes
│   └── 05_factors.ipynb         # content, correctness, confidence
├── annotations/              # committed: human-written labels and case notes
├── tests/
├── data/                     # gitignored: raw dataset
├── results/                  # scores.csv, tables/, figures/; maps/ is gitignored
├── report/                   # report source + final PDF
└── docs/                     # architecture, process, decisions, prompt log, scrum
```

## 3. Data flow

```
data/mit1003/ ──data.py──▶ images DataFrame + fixations + human maps
                                  │
                                  ▼
   scripts/compute_maps.py  (models.py, explain.py, device=mps|cuda|cpu)
        │                                   │
        ▼                                   ▼
 results/maps/*.npz (gitignored)    results/predictions.csv
        │                                   │
        ▼                                   │
 scripts/compute_scores.py (maps.py, alignment.py)
        ▼                                   │
 results/scores.csv  ◀── single source of truth ──┘
        ├───────────────────┬────────────────────────┐
        ▼                   ▼                        ▼
 evaluate.py (02_*)   failures.py (03_*)    annotations/*.csv (humans, blind)
 tables/*.csv         candidates, figures            │
        │                   ▼                        ▼
        │             run_probes.py (04_*)     factor tables (05_*)
        └───────────────────┴────────────────────────┘
                            ▼
                       report/ (PDF)
```

## 4. Spatial pipeline (D-03)

Heatmaps must live in one space before comparison, and the report must describe how they got there (spec §3). This is the most error-prone part of the project, so all of it lives in `maps.py`.

**Model input: resize the whole image to 224×224, no center crop.** The usual `Resize(256) + CenterCrop(224)` removes the image border, which would discard real fixations and make both maps artificially more central. The cost is a mild aspect-ratio distortion and possibly lower accuracy. A center-crop run on a subset measures the effect as a sensitivity check.

**Comparison canvas: 224×224 for every image.** No resampling is needed for 224-sized model maps, and because all maps share a shape, baselines that use other images' maps are trivial. The model grid (7×7 ResNet-50, 14×14 ViT) is far coarser than this canvas, so a larger one adds no information.

| Step | Human | Model |
|---|---|---|
| Source | Fixation points at native resolution | Grad-CAM or rollout on the model grid |
| Build | Gaussian blur at native resolution, σ = 24 px (D-13) | Bilinear upsample to 224×224 |
| To canvas | Area-downsample to 224×224; fixation coordinates scaled by (224/W, 224/H) | Already there |
| Thresholding | None | None |
| Per-metric normalization | CC: z-score | CC: z-score. NSS: z-score, sampled at fixations. sAUC: rank-based, none needed |

- Grad-CAM's ReLU can give an **all-zero map**. The score is then undefined: store `NaN`, count these per explainer, and report the count. Never add noise to hide them.
- Min-max scaling is for display only.
- Figures warp maps back to the original aspect ratio, so people are not shown squashed images.

## 5. Key interfaces

**`data.py`**: `load_images(root, n=None, seed=0) -> DataFrame` with columns `image_id, rel_path, width, height, n_fixations`. Also `load_fixations(image_id) -> (N,2) xy`, `load_fix_map(image_id)` and `load_image(image_id)`. Smoothing and scaling live in `maps.py`: `human_map_native(fixations, size, sigma)` and `scale_fixations(fixations, size)`.

**`maps.py`**: `model_input(image)`, `human_map_canvas(fixations, size, sigma)`, `model_map_canvas(grid)`, `scale_fixations(...)`, `zscore(map)` (all NaN if constant), `through_grid(map, cells)`, and for figures only `to_display(map)`, `warp_to_image(map, size)`. Resizing uses PIL on float32 arrays, so it is device-independent.

**`models.py`**: `ModelSpec(name, timm_id, family, target_layer, grid, reshape)`; `create(spec, device)` returns the model in eval mode; `preprocess(spec, image)` applies the whole-image resize and the model's **own** mean/std from the timm config (ResNet and ViT weights differ; never share one normalization); `predict(model, x, k)` returns the top-k classes. Weights are pinned in D-16.

**`explain.py`** (D-02): `ExplainerSpec(name, model, method, class_specific)`; `explain_grid(spec, model, x, target=None)` returns the raw map on the model's grid, and `explain(...)` returns it on the 224×224 canvas. `target` defaults to the model's top-1 prediction, because no ground-truth label exists here and the question is what supported the model's own decision.

| Explainer | Model | Class-specific | Purpose |
|---|---|---|---|
| `resnet50_gradcam` | ResNet-50 | Yes | CNN baseline |
| `vit_gradcam` | ViT-B/16 | Yes | Against ResNet: same method, different architecture |
| `vit_rollout` | ViT-B/16 | No | Against `vit_gradcam`: same model, different method |

The shared-model and shared-method pairs follow HW3's D-23 logic: without them, a difference could be either the architecture or the explanation method. Grad-CAM comes from the `grad-cam` package (spec §6 recommends existing implementations), stopping before its min-max scaling so all-zero maps stay visible (D-17). Rollout is about 20 lines over timm attention hooks, so it stays in-repo and tested.

**`alignment.py`** (D-04): `MetricSpec(name, fn, higher_is_better, kind, note)`, as in HW3's registry. `score(saliency, human, fixations, negatives)` returns all metrics for one map; `negatives` are the fixations of all other images (D-19).

| Metric | Measures | Why here |
|---|---|---|
| CC | Linear agreement between two maps | Symmetric, compares whole distributions, but depends on human-map blur |
| NSS | Model value at actual fixations | Uses raw fixations, so no blur parameter on the human side |
| sAUC | Separates this image's fixations from other images' fixations | Supplementary; discounts center bias, which CC and NSS reward |

**`scores.csv`**: tidy, one row per `(image_id, explainer)`. Columns `cc, nss, sauc, device`. Baselines are stored as extra `explainer` values, so all comparisons use the same code path.

**`predictions.csv`**: `image_id, model, top1_idx, top1_name, top1_prob, top5`.

**`evaluate.py`**: scores only mean something against reference points.

| Reference | Definition | Question it answers |
|---|---|---|
| `center` | Mean human map of all *other* images (leave-one-out) | Does the model beat "always look where people usually look"? |
| `shuffled` | This model map vs another image's human map, seeded permutation | How high is the score with no image-specific match? |
| `grid` (D-15) | Human map passed through the explainer's own grid (7×7 or 14×14) | What is the best score this grid allows? Mean CC ≈ 0.79 on 7×7, 0.94 on 14×14 |
| `ceiling` (stretch, §8) | Leave-one-subject-out human vs remaining humans | How high can any model get? |

Also: `summary()` for mean, median and image-bootstrap 95% CI; `paired()` for the Wilcoxon signed-rank test between explainers or against a baseline, on the same images; `factor_table()` for group comparisons and Spearman correlation with confidence.

**`failures.py`** (HW3 D-06 and D-24 pattern): quantitative first, humans choose last.
1. Reliability filter: drop images with too few fixations, plus low-ceiling images if `ceiling` exists.
2. Rank by percentile of CC and NSS combined, so one metric's scale cannot dominate.
3. Flag whether the image also falls below the `center` baseline.
4. `select_diverse(cands, k, score, max_per)` is reused from HW3 and caps candidates per explainer and scene type.
5. Write the full candidate list to `results/failures/`, then humans pick ≥ 5 from the top candidates and record why in `annotations/case_notes.csv`.

Cases are tagged **shared** (all explainers low) or **explainer-specific** (one low, another high), because these support different explanations. One or two high-alignment images are shown for contrast.

**`probes.py`** (D-07): a heatmap alone cannot show that a model relies on a region.
- `occlusion_probe()`: mask the top 20% of the human map, or of the model map, and record the drop in the target-class probability. If masking the model's region hurts confidence while masking the human region does not, the model is using something other than human-attended evidence. A shortcut claim in the report needs this support.
- `randomization_check()`: randomize model weights and measure how much the explanation changes. An explainer that barely changes is not reflecting what the model learned.

**`annotations/labels.csv`** (D-06): MIT1003 has no class labels, so "model correctness" requires human judgment. A seeded random 100 images, plus every case candidate, are labelled with `pred_quality` (`correct`, `plausible`, `wrong`, `no_applicable_class`) and `has_face`, `has_person`, `has_text`, `scene_type`, `dominant_object`. Annotators see only the image and the prediction, **never the heatmap or score**. Factor statistics use the random sample only; candidates are chosen for low scores and would bias them.

## 6. Runtime (cross-platform)

Same setup as HW3. The team roster is assumed to be unchanged; update this table if it isn't.

| | Member A | Member B |
|---|---|---|
| OS | macOS (Apple Silicon) | Windows + WSL2 Ubuntu |
| Device from `get_device()` | `mps` | `cuda` if available, else `cpu` |

- Code never hard-codes a device. Paths are `pathlib.Path` relative to the repo root, and CSVs store relative paths.
- Final numbers come from **one** full run on one machine (HW3 D-12). The other machine cross-checks a subset.
- `compute_maps.py` is resumable and supports `--limit N`.
- Seeds, device, package versions, and timm IDs go to `results/maps_run_info.json`.
- Committed: `scores.csv`, `predictions.csv`, `annotations/`. Gitignored: `data/`, `results/maps/`. Teammates can do all analysis without a GPU.

## 7. Correctness checks

Numbers are not used until these pass.

| # | Check |
|---|---|
| 1 | CC(a, a) = 1, CC(a, −a) = −1, CC is invariant to scale and offset. ✔ B-07 |
| 2 | NSS of a map peaked exactly at the fixations is high; a constant map gives `NaN`, not a crash. ✔ B-07 |
| 3 | sAUC ≈ 0.5 for a random map and for a pure center prior. ✔ B-07 |
| 4 | Our human map vs the dataset's own `fixMap`: high CC. ✔ B-03: all 1003 images, mean 0.99986, min 0.991 |
| 5 | Scaled fixations stay on the same content; a visual overlay in `01_data_check`. ✔ B-03: corner test + 5 images inspected |
| 6 | Grad-CAM: map changes with target class; rollout: a uniform-attention input gives a uniform map. ✔ B-06, plus a worked example for each and an independent Grad-CAM cross-check |
| 7 | Prediction sanity after the no-crop resize. ✔ B-05: 12 images inspected, 2 pinned in tests |
| 8 | Baseline ordering: `shuffled` scores are clearly below same-image human-vs-human scores |
| 9 | Cross-device: CPU vs MPS/CUDA maps agree closely on 20 images, tolerance recorded |
| 10 | `scores.csv` rows reproduce from the stored maps |

## 8. Proposed decisions

| ID | Proposal | Why, and alternatives |
|---|---|---|
| D-01 | **Dataset: MIT1003.** Fallback: CAT2000 train | Real eye-tracking on natural photos, the domain ImageNet models know, and small downloads (235 MB stimuli, 10 MB fixation maps, 47 MB per-subject data; URLs responded on 2026-11-09). SALICON has COCO labels but its attention comes from **mouse tracking**, a weaker claim about human vision. CAT2000 has useful categories, but many (fractal, jumbled, inverted) are far outside ImageNet |
| D-02 | ResNet-50 + ViT-B/16, three explainers (§5) | Separates architecture from method. Single-explainer is the minimum and would leave nothing to compare. CLIP is a stretch goal |
| D-03 | No-crop resize, 224×224 canvas (§4) | Keeps all fixations; center crop would bias alignment |
| D-04 | CC + NSS main, sAUC supplementary | Complementary, with a center-bias control. SIM and KL add little and cost report space |
| D-13 | Human blur σ = 24 native px (replaces D-05) | Best fit to the dataset's own fixation maps; its documentation gives no blur size or viewing geometry |
| D-06 | Score all 1003; annotate a random 100 blind | Far above the 50 minimum at little compute cost; hand labelling is the true cost, so it is capped |
| D-07 | Occlusion probe on selected cases | Turns "the model seems to use the background" into a testable claim |
| D-08 | Rollout: mean head fusion, no discard ratio | Closest to the original formulation; discard ratio is another tunable that invites cherry-picking |
| D-15 | Report each explainer's `grid` reference; also compare ResNet and ViT with the ViT map reduced to 7×7 | Coarse grids cap CC, so a direct ResNet vs ViT comparison partly measures grid size |

**Priority if time runs short.** Keep: D-01, ResNet and ViT with at least `resnet50_gradcam` and `vit_rollout`, CC + NSS, `center` and `shuffled`, 5 cases, annotations for the cases. Cut first: inter-observer `ceiling`, CLIP, then `randomization_check`. Center-crop sensitivity (B-25) stays, because cropping measurably changes predictions (D-16).

**`ceiling` depends on per-subject fixations.** Only `DATA.zip` has them, and it holds raw eye traces. Existing loaders such as `pysaliency` extract fixations by running Judd's MATLAB code, so they need MATLAB or Octave. This gets one time-boxed spike. If it fails, the missing ceiling goes in the report as a limitation.

## 9. Interpretation guardrails

Carried into `AGENTS.md`, as HW3 did for direction alignment.

1. **Task mismatch.** MIT1003 observers free-viewed the images; models were classifying them. Low alignment is not automatically a model failure, and the report must say so. The human maps also cover only each viewer's top 6 fixations (dataset README), so they describe **early** attention.
2. **No causal wording from heatmaps.** Write "the explanation highlights X", not "the model looks at X because…", unless a probe supports it.
3. **Center bias first.** A score is only reported as alignment alongside its `center` baseline.
4. **Humans own the interpretation.** AI drafts observations only; case explanations are the team's reasoning (course GenAI policy).
5. **Citations are checked** against the actual paper before entering the report.
