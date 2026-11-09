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

- [ ] **No-crop preprocessing has a cost (§2).** Compared with timm's default center crop on 100 images, the top-1 class agrees 66% (ResNet-50) and 55% (ViT-B/16), with nearly equal mean confidence. Many of these images have no clear ImageNet object.

- [ ] **Confidence is not comparable across models (§5).** Mean top-1 probability over all 1003 images is 0.209 for ResNet-50 and 0.461 for ViT-B/16, so confidence analysis must be done within each model.

- [ ] **ImageNet labels often do not fit (§1, §5).** Scene images get arbitrary labels (a misty lake as "traffic light"), which is why prediction quality is hand-labelled.

- [ ] **Hypothesis to test: Rollout hot spots on background (§4, §5).** On 6 inspected images, ViT Rollout showed bright isolated patches on empty background, and ViT Grad-CAM was speckled, unlike ResNet's smooth object-centered maps. The calculations pass their tests, so this looks like model behavior. Verify it on the full set in B-11/B-12 before claiming anything; a few extreme patches may also affect CC.

- [ ] **Main result: center prior beats every explainer on CC and NSS (§3).** Means over 1003 images, confidence intervals still to come (B-11):

  | Map | CC | NSS | sAUC |
  |---|---|---|---|
  | `resnet50_gradcam` | 0.288 | 0.917 | 0.580 |
  | `vit_gradcam` | 0.106 | 0.334 | 0.503 |
  | `vit_rollout` | 0.002 | 0.104 | 0.611 |
  | `center` | 0.422 | 1.326 | 0.502 |

  A map that ignores image content gets CC 0.422 against 0.288 for the best explainer.

- [ ] **Metrics rank the explainers differently (§3).** `vit_rollout` is lowest on CC and NSS but highest on sAUC. Its shuffled CC is negative (-0.098), which suggests it puts weight away from where people generally look, a pattern CC and NSS punish and sAUC discounts. Verify in B-11/B-12 before stating the cause.

- [ ] **Image-specific agreement is small (§3).** Score minus its own shuffled score, for CC / NSS / sAUC: ResNet-50 Grad-CAM +0.095 / +0.334 / +0.084; ViT Grad-CAM +0.030 / +0.097 / 0.000; ViT Rollout +0.100 / +0.401 / +0.117.

- [ ] **The two models often disagree (§1, §5).** Their top-1 class matches on only 47% of the 1003 images, which limits how far "the same task" can be assumed when comparing their explanations.

- [ ] **No all-zero maps.** None of the 3009 stored grids was all zero, so no scores are undefined for that reason (B-09).

- [ ] **Free viewing vs classification (§5 Discussion).** Human observers had no classification task, so low alignment is not automatically a model failure.
