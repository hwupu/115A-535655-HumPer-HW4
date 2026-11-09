# Human–Machine Attention Alignment on MIT1003

*[TEAM: add names and student IDs]*

## 1. Dataset and Model

**Human attention.** We use MIT1003 (Judd et al.), which has eye-tracking data from 15 observers free-viewing 1003 natural images. Its fixation maps cover each observer's top six fixations, so they describe early attention. We evaluate all 1003 images. MIT1003 has no class or content labels.

**Models and explanations.** We explain two pretrained ImageNet classifiers: ResNet-50 (He et al., `resnet50.tv2_in1k`) and ViT-B/16 (Dosovitskiy et al., `vit_base_patch16_224.augreg2_in21k_ft_in1k`), both from `timm` (Wightman). We use three explainers, always for the model's own top-1 class: Grad-CAM (Selvaraju et al.) on ResNet-50, Grad-CAM on ViT-B/16, and Attention Rollout (Abnar and Zuidema) on ViT-B/16. Comparing ResNet-50 with ViT Grad-CAM changes the architecture but keeps the method. Comparing ViT Grad-CAM with Rollout changes the method but keeps the model. Grad-CAM uses the `pytorch-grad-cam` package (Gildenblat et al.). We treat these maps as descriptions of what each method highlights, and not as a full account of model reasoning.

## 2. Experimental Setup

**Common space.** Every comparison uses a 224×224 canvas. The model receives the whole image resized to 224×224 with no center crop, because a crop removes real fixations. Human maps are made by blurring the fixation points with a Gaussian at native resolution and area-downsampling to the canvas. The dataset does not document its blur, so we fitted σ = 24 native pixels to its own fixation maps (mean CC 0.99986 across all 1003 images). Fixation coordinates are scaled to the canvas. Grad-CAM uses the target logit and ReLU; explanations are bilinearly upsampled from their 7×7 (ResNet-50) or 14×14 (ViT) grid. Rollout uses mean head fusion with no discard ratio. No thresholding is applied, and constant maps would be recorded as undefined (none occurred).

**Metrics.** CC compares whole maps. NSS reads the z-scored explanation at the actual fixations, so it does not depend on the human-map blur. Shuffled AUC (sAUC; Zhang et al.) uses fixations from all other images as negatives. Human fixations favor the image center (Tatler), and CC and NSS reward a map for matching that general tendency, whereas sAUC penalizes center bias (Bylinskii et al.). Using all three separates agreement about a particular image from agreement about where people usually look.

**Reference maps.** A score alone is hard to interpret, so each explainer is compared with: (1) a *center prior*, the mean human map of all other images; (2) a *shuffled* pairing of each explanation with another image's human data; and (3) a *grid limit*, the human map passed through the explainer's own grid. The grid limit matters because a coarse grid cannot represent a detailed map: it caps mean CC at about 0.79 on 7×7 and 0.94 on 14×14. To compare models at equal resolution, we also reduce ViT explanations to 7×7. Intervals are 95% image-level bootstrap intervals, and paired tests are Holm-corrected Wilcoxon signed-rank tests.

**Reproducibility.** Code, scores and scripts are in our repository. Reported numbers come from one full run on Apple MPS. On 20 images, CPU and MPS predictions were identical and scores differed by at most 0.00002.

## 3. Quantitative Alignment

| Map | CC | NSS | sAUC |
|---|---|---|---|
| ResNet-50 Grad-CAM | 0.288 (0.276, 0.300) | 0.917 (0.874, 0.960) | 0.580 (0.571, 0.589) |
| ViT Grad-CAM | 0.106 (0.093, 0.119) | 0.334 (0.285, 0.382) | 0.503 (0.494, 0.512) |
| ViT Rollout | 0.002 (−0.010, 0.014) | 0.104 (0.058, 0.152) | 0.611 (0.602, 0.621) |
| Center prior | 0.422 (0.413, 0.433) | 1.326 (1.293, 1.363) | 0.502 (0.495, 0.510) |
| Grid limit, 7×7 | 0.786 | 2.616 | 0.800 |
| Grid limit, 14×14 | 0.941 | 3.408 | 0.862 |

*Table 1. Mean alignment over 1003 images, with 95% intervals.*

![Mean alignment for explanations and references](../results/figures/alignment/alignment_means.png)

*Figure 1. Mean alignment with 95% intervals. Dotted lines mark 0 for CC and NSS and chance (0.5) for sAUC.*

**A center prior beats every explanation on CC and NSS.** A map that ignores the image reaches CC 0.422, against 0.288 for the best explanation (paired difference −0.135, Holm p < 0.001). On sAUC, which discounts center bias, the center prior is at chance and ResNet-50 Grad-CAM (0.580) and Rollout (0.611) are above it. ViT Grad-CAM (0.503) is not distinguishable from the center prior or from its own shuffled pairing on sAUC (Holm p = 0.99).

**The metrics rank explainers differently.** Rollout is lowest on CC and NSS and highest on sAUC. Its maps correlate negatively with the center prior (mean CC −0.221, negative on 82% of images), compared with +0.462 for ResNet-50 Grad-CAM. This placement away from the center is penalized by CC and NSS and discounted by sAUC.

**Grid size does not explain the gap between models.** ViT Grad-CAM reduced to 7×7 reaches CC 0.141, still below ResNet-50 Grad-CAM (difference 0.147, Holm p < 0.001).

**Alignment varies by image category.** Categories come from the image or human data only: faces (YuNet; Wu et al.), text (EasyOCR, using CRAFT; Baek et al.), small salient attention, edge/texture-rich images, and peripheral attention. Table 2 compares each explainer inside and outside a category on its own CC + NSS percentile, so the comparison is relative to that explainer's overall performance.

| Explainer | Category | Inside − outside | n inside |
|---|---|---|---|
| ViT Rollout | Text | −0.189 | 180 |
| ResNet-50 Grad-CAM | Faces | −0.176 | 274 |
| ViT Grad-CAM | Faces | −0.128 | 274 |
| ViT Rollout | Edge/texture-rich | −0.107 | 201 |
| ResNet-50 Grad-CAM | Text | −0.096 | 180 |
| ResNet-50 Grad-CAM | Edge/texture-rich | −0.082 | 201 |
| ViT Rollout | Small salient | +0.184 | 201 |

*Table 2. Category effects that survive Holm correction across 15 tests. Rollout on faces is +0.008 and not significant. Effects are associations. On a visual check of 20 sampled images per detector, detected faces looked correct and about three-quarters of text flags showed real text.*

**Occlusion.** We masked the top 20% of each map and measured the drop in the probability of the top-1 class (Table 3). Grad-CAM regions lower confidence more than human-attended regions, and human-attended regions lower it more than random regions of equal area. Rollout regions are close to random and below human regions, despite Rollout's high sAUC. As a sanity check (Adebayo et al.), maps from randomly initialized models were unrelated to trained maps (mean Spearman within ±0.021, 100 images).

| Model | Human region | Grad-CAM region | Rollout region | Random |
|---|---|---|---|---|
| ResNet-50 | 0.122 / 0.117 | 0.147 / 0.142 | n/a | 0.084 / 0.087 |
| ViT-B/16 | 0.240 / 0.246 | 0.333 / 0.341 | 0.177 / 0.173 | 0.152 / 0.177 |

*Table 3. Mean drop in top-1 probability, blur fill / mean-color fill, 1003 images.*

**Sensitivity to center cropping.** On 200 images with Resize(256) and CenterCrop(224), which keeps 89.8% of fixations on average, the three explainers keep their order on every metric and the center prior stays first on CC and NSS.

## 4. Misalignment Cases

**[TEAM TO WRITE: the explanation for each case. Everything below is evidence, not explanation.]**

We chose cases by metric, not by appearance: for each Holm-significant category effect, the image with the largest gap between the two explainers, excluding the 5% of images with the fewest fixations.

![The five cases](../results/figures/failures/report_cases.png)

*Figure 2. Input, human attention, and the three explanations, with each model's top-1 class and scores. Maps are scaled for display.*

| # | Category | Weak → better fit | Predicted class (weak model) | Weak: CC / NSS / sAUC | Better fit: CC / NSS / sAUC | Occlusion drop: own / human / random (blur; mean) | Baseline prob. |
|---|---|---|---|---|---|---|---|
| 1 | Faces | ResNet Grad-CAM → Rollout | mountain bike | −0.10 / −0.38 / 0.22 | 0.40 / 1.66 / 0.73 | 0.309 / 0.204 / 0.197; 0.309 / 0.284 / 0.232 | 0.313 |
| 2 | Text | Rollout → ViT Grad-CAM | menu | −0.34 / −1.38 / 0.21 | 0.53 / 2.15 / 0.78 | 0.071 / 0.223 / 0.120; 0.030 / 0.222 / 0.150 | 0.223 |
| 3 | Edge/texture | Rollout → ViT Grad-CAM | mask | −0.19 / −0.56 / 0.42 | 0.49 / 1.45 / 0.69 | −0.058 / 0.581 / 0.079; −0.077 / 0.695 / 0.112 | 0.746 |
| 4 | Small salient | ViT Grad-CAM → Rollout | lakeside | −0.16 / −0.65 / 0.26 | 0.34 / 1.48 / 0.84 | 0.286 / −0.233 / −0.009; 0.112 / 0.100 / 0.126 | 0.602 |
| 5 | Faces | ViT Grad-CAM → Rollout | stage | −0.23 / −0.87 / 0.22 | 0.27 / 1.01 / 0.81 | 0.147 / −0.273 / −0.027; 0.199 / −0.342 / −0.078 | 0.333 |

*Table 4. Case evidence. Occlusion values are for the weak explainer's model. For case 4 the human-mask result changes sign between fills, so occlusion is not reliable there.*

For each case, explain what the human and the model appear to prioritize, and why. Possible reasons from the brief: background context, object texture, edges, small salient objects, faces, text, contextual shortcuts, or task-irrelevant regions. Note whether it suggests a shortcut (Geirhos et al.) or a failure of perceptual alignment.

- **Case 1:** [TEAM]
- **Case 2:** [TEAM]
- **Case 3:** [TEAM]
- **Case 4:** [TEAM]
- **Case 5:** [TEAM]

## 5. Discussion

**[TEAM TO WRITE, using the evidence below. Answer: is the model using visual evidence that is consistent with human attention?]**

Evidence available:

- **Content:** Table 2. Faces lower both Grad-CAM maps but not Rollout.
- **Model correctness (proxies only):** higher confidence goes with slightly better CC and NSS for ResNet-50 Grad-CAM (rho 0.12, 0.11) and ViT Grad-CAM (0.19, 0.17), and not reliably for Rollout. Where ResNet-50 and ViT agree on the top-1 class (47% of images), ViT Grad-CAM aligns better (+0.054, Holm p = 0.010).
- **Salient objects / attention spread:** broader human attention goes with worse Rollout scores (CC rho −0.44, NSS −0.38, sAUC −0.30).
- **Plausible maps can mislead:** Rollout has the highest sAUC, but masking its regions is close to masking random regions (Table 3).
- **Limitations to choose from:** free-viewing observers had no classification task; the maps show early fixations only; there are no labels for correctness or content; detectors are imperfect; occlusion produces unnatural images; ImageNet classes fit many scenes poorly, and the two models often explain different predictions.
- **Possible follow-up experiments:** task-driven eye-tracking on the same images; adding labels for correctness and content; comparing other explainers or a CLIP encoder.

## Contribution Statement

*[TEAM TO WRITE: each member's role. Excluded from the page limit.]*

## Works Cited

Abnar, Samira, and Willem Zuidema. "Quantifying Attention Flow in Transformers." *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 2020, pp. 4190–97, https://doi.org/10.18653/v1/2020.acl-main.385.

Adebayo, Julius, et al. "Sanity Checks for Saliency Maps." *arXiv*, 2018, https://doi.org/10.48550/arXiv.1810.03292.

Baek, Youngmin, et al. "Character Region Awareness for Text Detection." *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2019, pp. 9357–66, https://doi.org/10.1109/CVPR.2019.00959.

Bylinskii, Zoya, et al. "What Do Different Evaluation Metrics Tell Us About Saliency Models?" *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 41, no. 3, 2019, pp. 740–57, https://doi.org/10.1109/TPAMI.2018.2815601.

Dosovitskiy, Alexey, et al. "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale." *arXiv*, 2020, https://doi.org/10.48550/arXiv.2010.11929.

Geirhos, Robert, et al. "Shortcut Learning in Deep Neural Networks." *Nature Machine Intelligence*, vol. 2, no. 11, 2020, pp. 665–73, https://doi.org/10.1038/s42256-020-00257-z.

Gildenblat, Jacob, et al. *PyTorch Library for CAM Methods*. *GitHub*, github.com/jacobgil/pytorch-grad-cam.

He, Kaiming, et al. "Deep Residual Learning for Image Recognition." *2016 IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 770–78, https://doi.org/10.1109/CVPR.2016.90.

Jaided AI. *EasyOCR*. *GitHub*, github.com/JaidedAI/EasyOCR.

Judd, Tilke, et al. "Learning to Predict Where Humans Look." *2009 IEEE 12th International Conference on Computer Vision*, 2009, pp. 2106–13, https://doi.org/10.1109/ICCV.2009.5459462.

Selvaraju, Ramprasaath R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." *International Journal of Computer Vision*, vol. 128, no. 2, 2020, pp. 336–59, https://doi.org/10.1007/s11263-019-01228-7.

Tatler, Benjamin W. "The Central Fixation Bias in Scene Viewing: Selecting an Optimal Viewing Position Independently of Motor Biases and Image Feature Distributions." *Journal of Vision*, vol. 7, no. 14, 2007, p. 4, https://doi.org/10.1167/7.14.4.

Wightman, Ross. *PyTorch Image Models*. *GitHub*, github.com/huggingface/pytorch-image-models.

Wu, Wei, et al. "YuNet: A Tiny Millisecond-Level Face Detector." *Machine Intelligence Research*, vol. 20, no. 5, 2023, pp. 656–65, https://doi.org/10.1007/s11633-023-1423-y.

Zhang, Lingyun, et al. "SUN: A Bayesian Framework for Saliency Using Natural Statistics." *Journal of Vision*, vol. 8, no. 7, 2008, p. 32, https://doi.org/10.1167/8.7.32.
