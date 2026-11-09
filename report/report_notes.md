# Notes for the report

Points that must reach the report. Check off when drafted.

## Must include

- [ ] **Grid size limits the score (D-15, §2 Experimental Setup and §3 Quantitative).**
  - Problem: ResNet-50 explanations come from a 7×7 grid and ViT-B/16 explanations from 14×14. A coarse grid cannot represent a detailed map, so a score partly reflects resolution. Comparing raw scores across models would favor ViT regardless of how human-like its evidence is.
  - Evidence: passing all 1003 human maps through each grid (area-downsample, then bilinear upsample) limits mean CC to **0.786 on 7×7** and **0.941 on 14×14**. These are estimates from one resampling path.
  - Our solution: (1) report each explainer next to its `grid` reference, the score of the human map passed through that explainer's own grid; (2) also compare ResNet and ViT with the ViT map reduced to 7×7, so both have the same resolution.
  - Final numbers (B-10, 1003 images): `grid_7x7` CC 0.786 and `grid_14x14` CC 0.941, confirming the estimates.
  - Outcome: reducing ViT to 7×7 did not lower its score. `vit_gradcam` CC went from 0.106 to 0.141, and ResNet-50 Grad-CAM (0.288) stays ahead at equal resolution. So grid size limits what is possible, but it does not explain the gap between the models.

- [ ] **Human maps describe early attention (§1 Dataset).** They cover each viewer's top 6 fixations (dataset README).

- [ ] **Human blur (§2).** σ = 24 native px, fitted to the dataset's own maps because its documentation gives no blur size or viewing geometry (D-13).

- [ ] **Center-crop sensitivity (§2; D-25).** On 200 images, the crop keeps 89.8% of fixations on average (lowest 40.4%). The order of explainers and center prior stays the same on each metric. Under the crop, center-prior CC is 0.338 (no crop 0.403) and NSS 0.812 (1.262); ResNet-50 Grad-CAM NSS is 0.627 (0.901). The center prior still leads on CC and NSS. Do not say the crop inflates alignment. Table: `results/tables/crop_sensitivity.csv`.

- [ ] **No-crop preprocessing has a cost (§2).** Compared with timm's default center crop on 100 images, the top-1 class agrees 66% (ResNet-50) and 55% (ViT-B/16), with nearly equal mean confidence. Many of these images have no clear ImageNet object.

- [ ] **Confidence is not comparable across models (§5).** Mean top-1 probability over all 1003 images is 0.209 for ResNet-50 and 0.461 for ViT-B/16, so confidence analysis must be done within each model.

- [ ] **ImageNet labels often do not fit (§1, §5).** Scene images get arbitrary labels (a misty lake as "traffic light"), so "correct" is not well defined for many images.

- [ ] **Hypothesis to test: Rollout hot spots on background (§4, §5).** On 6 inspected images, ViT Rollout showed bright isolated patches on empty background, and ViT Grad-CAM was speckled, unlike ResNet's smooth object-centered maps. The calculations pass their tests, so this looks like model behavior. Verify it on the full set in B-11/B-12 before claiming anything; a few extreme patches may also affect CC.

- [ ] **Main result: center prior beats every explainer on CC and NSS (§3).** Means over 1003 images, with intervals (95% bootstrap intervals in `results/tables/alignment_summary.csv`, tests in `paired_tests.csv`, figure in `results/figures/alignment/alignment_means.png`):

  | Map | CC | NSS | sAUC |
  |---|---|---|---|
  | `resnet50_gradcam` | 0.288 | 0.917 | 0.580 |
  | `vit_gradcam` | 0.106 | 0.334 | 0.503 |
  | `vit_rollout` | 0.002 | 0.104 | 0.611 |
  | `center` | 0.422 | 1.326 | 0.502 |

  A map that ignores image content gets CC 0.422 against 0.288 for the best explainer.

- [ ] **Metrics rank the explainers differently (§3).** `vit_rollout` is lowest on CC and NSS but highest on sAUC. Its shuffled CC is negative (-0.098). B-11 confirms the spatial pattern: Rollout maps correlate with the center prior at mean CC -0.221 and are negative on 82% of images, against +0.462 for ResNet-50 Grad-CAM and +0.178 for ViT Grad-CAM. So CC and NSS punish its off-center weight, while sAUC discounts that. This describes where the map puts weight, and does not by itself show why.

- [ ] **Image-specific agreement is small (§3).** Score minus its own shuffled score, for CC / NSS / sAUC: ResNet-50 Grad-CAM +0.095 / +0.334 / +0.084; ViT Grad-CAM +0.030 / +0.097 / 0.000; ViT Rollout +0.100 / +0.401 / +0.117. After Holm correction across 27 tests, all are significant except ViT Grad-CAM on sAUC (p = 0.99), which shows no image-specific agreement on that metric. With 1003 images, report effect sizes as well as p-values.

- [ ] **The two models often disagree (§1, §5).** Their top-1 class matches on only 47% of the 1003 images, which limits how far "the same task" can be assumed when comparing their explanations.

- [ ] **No all-zero maps.** None of the 3009 stored grids was all zero, so no scores are undefined for that reason (B-09).

- [ ] **Limitation: no labels for correctness or content (§5).** MIT1003 has no class or content labels and we did not hand-label images (D-21). Confidence and model agreement are proxies for correctness. Claims about faces, text and scene content come from the final cases only. A natural follow-up is to add such labels and measure these effects across images.

- [ ] **Category ranking (§3, §4; D-22).** Inside minus outside a category on each explainer's own CC + NSS percentile, n = 1003, Holm across 15 tests. Table: `results/tables/category_effects.csv`.
  - Harder than the explainer's own average: ViT Rollout on text (-0.189), ResNet-50 Grad-CAM on faces (-0.176), ViT Grad-CAM on faces (-0.128), ViT Rollout on edge/texture-rich images (-0.107), ResNet-50 Grad-CAM on text (-0.096) and on edge/texture-rich images (-0.082).
  - Easier: ViT Rollout on small salient attention (+0.184).
  - Rollout on faces is about average (+0.008), so faces hurt the Grad-CAM maps specifically.
  - Peripheral human attention shows no effect that survives correction.
  - Say "associated with", not "caused by". Categories overlap, and the percentile is relative to each explainer.

- [ ] **Detector limits (§2).** Face detections were all correct on 20 sampled images. Text flags were real on about three-quarters of 20 sampled images, with false hits such as ornate ceilings, so text effects are less certain. Both are one-off visual checks.

- [ ] **Occlusion across all images (§3, §5; D-23).** Mean drop in target-class probability when masking 20% of the image, blur / mean-color fill. `results/tables/occlusion_summary.csv`, `occlusion_paired.csv`.

  | Model | Human region | Own Grad-CAM | ViT Rollout | Random |
  |---|---|---|---|---|
  | ResNet-50 | 0.122 / 0.117 | 0.147 / 0.142 | n/a | 0.084 / 0.087 |
  | ViT-B/16 | 0.240 / 0.246 | 0.333 / 0.341 | 0.177 / 0.173 | 0.152 / 0.177 |

  - Grad-CAM regions hurt confidence more than human-attended regions, in both models and fills. Masking is only one way to probe reliance, and masked images are unnatural.
  - Human-attended regions also matter: they exceed random masks, so human and model evidence are partly shared.
  - Rollout regions are close to random (above it with blur, level with it with mean fill) and below human regions, even though Rollout has the best sAUC. A high human-alignment score did not mean the map found what the model depends on. This fits the spec's warning about attention maps.

- [ ] **Weight randomization (§2, §5).** Mean Spearman between trained and random-weight maps on 100 images: ResNet-50 Grad-CAM 0.015, ViT Grad-CAM -0.005, ViT Rollout -0.021, so all three depend on learned weights. Some random-weight Grad-CAM maps were all zero and are excluded (77 and 92 valid of 100).

- [ ] **Case occlusion (§4).** `results/tables/occlusion_cases.csv`. Case 4 (willow) changes sign between fills for the human mask, so do not lean on occlusion there.

- [ ] **Observable factors (§5; D-21).** All associations, with n = 1003 and Holm correction. Tables: `factor_correlations.csv`, `model_agreement_effect.csv`.
  - Confidence: higher confidence goes with slightly better alignment on CC and NSS for ResNet-50 Grad-CAM (rho 0.12, 0.11) and ViT Grad-CAM (0.19, 0.17), but not reliably for ViT Rollout. Effects are small and not significant on sAUC.
  - Spread of human attention: broader attention goes with worse ViT Rollout scores (CC rho -0.44, NSS -0.38, sAUC -0.30) and worse ResNet-50 Grad-CAM on NSS and sAUC (-0.22, -0.20). ViT Grad-CAM is unaffected.
  - Model agreement: images where ResNet-50 and ViT agree align better for ViT Grad-CAM (+0.054, Holm p = 0.010). ResNet-50 Grad-CAM (+0.037) and ViT Rollout (+0.004) do not survive correction.

- [ ] **Free viewing vs classification (§5 Discussion).** Human observers had no classification task, so low alignment is not automatically a model failure.
