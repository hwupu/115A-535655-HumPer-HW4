# Notes for the report

Points that must reach the report. Check off when drafted.

## Must include

- [ ] **Grid size limits the score (D-15, §2 Experimental Setup and §3 Quantitative).**
  - Problem: ResNet-50 explanations come from a 7×7 grid and ViT-B/16 explanations from 14×14. A coarse grid cannot represent a detailed map, so a score partly reflects resolution. Comparing raw scores across models would favor ViT regardless of how human-like its evidence is.
  - Evidence: passing all 1003 human maps through each grid (area-downsample, then bilinear upsample) limits mean CC to **0.786 on 7×7** and **0.941 on 14×14**. These are estimates from one resampling path.
  - Our solution: (1) report each explainer next to its `grid` reference, the score of the human map passed through that explainer's own grid; (2) also compare ResNet and ViT with the ViT map reduced to 7×7, so both have the same resolution.
  - Use the final numbers from B-10, not only the estimates above.

- [ ] **Human maps describe early attention (§1 Dataset).** They cover each viewer's top 6 fixations (dataset README).

- [ ] **Human blur (§2).** σ = 24 native px, fitted to the dataset's own maps because its documentation gives no blur size or viewing geometry (D-13).

- [ ] **No-crop preprocessing has a cost (§2).** Compared with timm's default center crop on 100 images, the top-1 class agrees 66% (ResNet-50) and 55% (ViT-B/16), with nearly equal mean confidence. Many of these images have no clear ImageNet object.

- [ ] **Confidence is not comparable across models (§5).** Mean top-1 probability on those images is about 0.19 for ResNet-50 and 0.41 for ViT-B/16, so confidence analysis must be done within each model.

- [ ] **ImageNet labels often do not fit (§1, §5).** Scene images get arbitrary labels (a misty lake as "traffic light"), which is why prediction quality is hand-labelled.

- [ ] **Hypothesis to test: Rollout hot spots on background (§4, §5).** On 6 inspected images, ViT Rollout showed bright isolated patches on empty background, and ViT Grad-CAM was speckled, unlike ResNet's smooth object-centered maps. The calculations pass their tests, so this looks like model behavior. Verify it on the full set in B-11/B-12 before claiming anything; a few extreme patches may also affect CC.

- [ ] **Free viewing vs classification (§5 Discussion).** Human observers had no classification task, so low alignment is not automatically a model failure.
