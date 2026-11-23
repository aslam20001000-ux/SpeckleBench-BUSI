# Final Analysis: AI-Based vs. Traditional Speckle Reduction on BUSI

All numbers in this section come from the executed experiment (`results/report.md`, `results/per_image_results.csv`). No value is estimated or taken from the literature.

## 1. Experimental conditions (summary)

- **Data:** 15 BUSI test images, stratified across benign, malignant and normal, converted to grayscale and resized to 128×128.
  - 10 validation images were used for parameter selection.
  - 200 training images were disjoint from the test set.
- **Noise:** synthetic multiplicative speckle, y = x·(1 + σ·n) with n ~ N(0,1), at σ ∈ {0.1, 0.2, 0.3}.
  - Every method received exactly the same noisy images (fixed seeds).
- **Reference:** the original BUSI image. It already contains native speckle (mean ENL of the reference ROIs = 58.6), so PSNR and SSIM measure removal of the *added* noise.
- **Tuning:** every method's free parameters were chosen on the validation split only. This covers the classical filter parameters, the DRUNet noise-level input and the DIP iteration count.
- **Hardware:** all eight methods ran on the same CPU (Intel Core i5-10210U, 4 cores, no GPU). Processing times are therefore directly comparable.
- **Ranking criterion (defined before the run):** mean per-image rank across PSNR, SSIM and EPI. ENL is reported but not ranked, because it also rewards over-smoothing.

| Method | Type | Training performed in this project |
|---|---|---|
| DnCNN | AI, supervised | Trained from scratch: 1,500 iterations, 33.3 min |
| Noise2Void | AI, self-supervised | Trained from scratch on noisy images only: 1,500 iterations, 9.6 min |
| DRUNet | AI, pretrained | None. deepinv grayscale weights, used zero-shot in the log domain |
| Deep Image Prior (DIP) | AI, untrained | None. Optimised per image; iteration count chosen on validation (17.7 min) |
| Median, Bilateral, Lee, NLM | Traditional | None. Parameters chosen on validation |

## 2. Overall results

**Table 1: Mean over all noise levels and test images.**

| Method | PSNR (dB) | SSIM | EPI | ENL | Time / image | Mean rank ↓ |
|---|---|---|---|---|---|---|
| **DnCNN** | 29.35 | 0.854 | **0.912** | 56.4 | 191 ms | **1.82** |
| DRUNet | **29.37** | **0.856** | 0.902 | 70.9 | 637 ms | 1.96 |
| **Lee** | 28.34 | 0.838 | 0.886 | 55.0 | 0.7 ms | 3.62 |
| Bilateral | 28.33 | 0.828 | 0.878 | 69.5 | **0.6 ms** | 4.04 |
| Noise2Void | 27.40 | 0.795 | 0.892 | 67.5 | 64 ms | 4.97 |
| DIP | 27.63 | 0.771 | 0.877 | 65.5 | 103.9 s | 5.53 |
| Median | 26.59 | 0.768 | 0.827 | 62.6 | 13.8 ms | 6.90 |
| NLM | 26.17 | 0.711 | 0.815 | **144.0** | 44 ms | 7.16 |

**Table 2: Per noise level (mean over 15 images).**

| σ | Noisy input PSNR / SSIM | Best AI (PSNR / SSIM) | Best traditional (PSNR / SSIM) |
|---|---|---|---|
| 0.1 | 28.75 / 0.880 | DRUNet 33.26 / 0.943 | Lee 31.34 / 0.923 |
| 0.2 | 22.85 / 0.695 | DRUNet 29.30 / 0.873 | Bilateral 28.11 / 0.843 |
| 0.3 | 19.43 / 0.545 | DnCNN 26.79 / 0.776 | Bilateral 25.86 (PSNR) / Lee 0.760 (SSIM) |

Figures: `results/figures/metrics_bars.png`, `psnr_ssim_vs_noise.png`, `psnr_boxplots.png`, `ssim_boxplots.png`, `processing_time.png`, `visual_sigma*.png`.

## 3. Best-performing AI model

**DnCNN is the best AI model** under the pre-defined criterion (mean rank 1.82). The margin over DRUNet (1.96) is small, and the two models dominate at different noise levels:

- **At low and moderate noise (σ = 0.1, 0.2), DRUNet is better than DnCNN.** It wins on PSNR, SSIM and EPI on almost every image. At σ = 0.1 it wins on all 15 images, with a median PSNR difference of +0.90 dB (Wilcoxon p = 1×10⁻⁴).
- **At high noise (σ = 0.3), DnCNN is clearly better.**
  - It wins on PSNR (+1.25 dB median, 15/15 images) and EPI (15/15 images).
  - All nine DnCNN–DRUNet comparisons remain significant after Holm correction (largest adjusted p = 0.035).
- **Why DRUNet drops at σ = 0.3:** DRUNet was pretrained for additive Gaussian noise. It was applied here through a log transform, which turns multiplicative speckle into additive noise. At high σ, however, the log-domain noise becomes strongly skewed, with large negative excursions and clipped zero pixels.
  - The visual results at σ = 0.3 show isolated **black-dot artefacts** in DRUNet outputs (`visual_sigma0.3.png`).
  - This is consistent with outliers that a Gaussian-trained denoiser does not remove.
- **Why DnCNN is robust:** it was trained directly on the multiplicative noise model across the whole range σ ∈ [0.05, 0.35], so its performance degrades smoothly.

**Conclusion:** DnCNN is the most reliable AI model across the full noise range. DRUNet is the best choice when noise is low to moderate and no training is possible.

## 4. Best traditional filter

**The Lee filter is the best traditional filter** (mean rank 3.62), closely followed by the Bilateral filter (4.04).

- Lee has the highest traditional SSIM at σ = 0.1 and σ = 0.3, and the highest traditional EPI at every level.
- Bilateral has a slightly higher PSNR at σ = 0.2 and σ = 0.3.
- Lee's advantage comes from its speckle model. It uses local mean and variance to estimate how much smoothing each pixel needs, smoothing homogeneous tissue strongly while leaving high-variance regions (edges, lesion boundaries) mostly untouched.
- **Median and NLM performed worst.**
  - At σ = 0.1, Median gave no PSNR gain over the noisy input (−0.01 dB) and lowered SSIM (−0.026).
  - NLM produced the highest ENL of all methods (144.0, far above the reference's 58.6) but the lowest SSIM (0.711) and EPI (0.815). This means it smooths away genuine tissue texture and fine structure, not only noise.

## 5. AI-based vs. traditional methods

### 5.1 Image quality (PSNR, SSIM)

- **The best AI models outperform the best traditional filter.**
  - DnCNN beat Lee on PSNR at every noise level, by a median of about +1.0 dB on all 15/15 images (Holm-adjusted p = 0.003 at each level).
  - DnCNN's SSIM advantage over Lee is significant at σ = 0.1 and 0.2 but not at σ = 0.3 (adjusted p = 1.0).
- **As groups, the AI methods also score higher** at every noise level:

| σ | AI mean PSNR / SSIM | Traditional mean PSNR / SSIM |
|---|---|---|
| 0.1 | 31.29 / 0.899 | 30.20 / 0.882 |
| 0.2 | 28.18 / 0.823 | 26.97 / 0.787 |
| 0.3 | 25.85 / 0.735 | 24.90 / 0.690 |

- **"AI" is not uniformly better.**
  - Noise2Void and DIP rank below Lee and Bilateral overall.
  - At σ = 0.1, Noise2Void lowered SSIM relative to the noisy input (−0.039), and DIP did not improve it (±0.000).

### 5.2 Edge preservation (EPI)

- **Edge preservation is where the AI advantage is largest at high noise.**
  - At σ = 0.3, DnCNN reached EPI 0.852 against Lee's 0.800, winning on all 15/15 images (adjusted p = 0.003).
  - Noise2Void (0.837) also beat Lee on all 15 images at σ = 0.3 (adjusted p = 0.003).
- **Traditional filters lose edges as noise increases.** Their mean EPI dropped from 0.953 (σ = 0.1) to 0.738 (σ = 0.3). Median (0.679) and NLM (0.690) dropped the most.
- **The AI group dropped less**, from 0.960 to 0.825.
- **Lesion boundaries:** the visual comparisons show sharper lesion boundaries for DnCNN than for Median, Bilateral and NLM, which visibly blur the boundary at σ = 0.3.

### 5.3 Noise reduction (ENL)

- **All methods increased ENL substantially over the noisy input** (33.5, 15.9 and 9.2 at σ = 0.1, 0.2, 0.3).
- **ENL must be read together with SSIM and EPI.** An ENL far above the reference's own value (58.6) indicates smoothing beyond the original tissue texture, not better denoising.
  - **NLM:** ENL of 144 combined with the lowest SSIM is the clearest example.
  - **DnCNN (56.4) and Lee (55.0):** ENL values closest to the reference, meaning they restore homogeneous regions to roughly their original appearance.
  - **DRUNet (70.9):** smooths homogeneous regions more strongly. Its ENL is significantly higher than Lee's at all three noise levels (adjusted p ≤ 0.038).
  - **DnCNN vs Lee:** the ENL difference was not significant at any level.

### 5.4 Computational cost

All times were measured on the same CPU, per 128×128 image.

| Method | Inference time | One-off cost |
|---|---|---|
| Bilateral / Lee | 0.6–0.7 ms | none |
| Median / NLM | 14 / 44 ms | none |
| Noise2Void | 64 ms | 9.6 min training |
| DnCNN | 191 ms | 33.3 min training |
| DRUNet | 637 ms | 32.6 M parameters, 124 MB weights |
| DIP | 103.9 s (74–135 s) | none, but cost is paid for every image |

- **Traditional filters are roughly 300–1,000× faster** than the best AI models. Lee gives about 96% of DnCNN's overall PSNR at under 0.4% of its inference time.
- **DnCNN and DRUNet are still well within interactive speed** on a laptop CPU, and would be much faster on a GPU.
- **DIP is impractical for routine use.** Optimising a network for every image takes over a minute per frame, and its quality is still below the Lee filter.

## 6. Advantages and disadvantages of each method

| Method | Advantages (observed) | Disadvantages (observed) |
|---|---|---|
| **DnCNN** | Best overall rank. Most robust at high noise. Best edge preservation at σ = 0.3. ENL close to the reference (no over-smoothing). | Needs paired clean/noisy training data. Longest training (33 min on CPU). Learns the synthetic noise model, so transfer to real scanner speckle is not guaranteed. |
| **DRUNet** | Best quality at σ = 0.1–0.2 with **no training at all**. Pretrained weights were verified at runtime (+8.46 dB on a Gaussian check). | Domain mismatch (trained for Gaussian noise on natural images). Black-dot artefacts and a quality drop at σ = 0.3. Largest model (32.6 M parameters). Slowest feed-forward method (0.64 s/image). |
| **Noise2Void** | Needs **no clean images**, so it could be trained on real clinical scans. Fast inference. Strong edge preservation at high noise. | Weakest AI model at low noise; it reduced SSIM at σ = 0.1. Assumes pixel-independent noise, which real ultrasound speckle violates. |
| **DIP** | Needs no training data at all; adapts to each image. | Extremely slow (~104 s/image). Below the Lee filter on SSIM at every level. Sensitive to the stopping iteration, which had to be tuned. |
| **Lee** | Best traditional filter. Designed for multiplicative noise. Sub-millisecond speed. Good structure preservation. No training. | About 1 dB below DnCNN in PSNR. Edge preservation drops at high noise (EPI 0.800 at σ = 0.3). Needs the noise level parameter (C_u) tuned. |
| **Bilateral** | Fastest method (0.6 ms). Close second among traditional filters. | Assumes additive noise. Lower SSIM than Lee. Over-smooths at high noise. |
| **Median** | Simple and robust to impulse outliers. | No improvement at σ = 0.1. Weak edge preservation at high noise (EPI 0.679 at σ = 0.3). |
| **NLM** | Strongest smoothing of homogeneous regions. | Lowest SSIM and EPI. Over-smooths texture (ENL 144 vs reference 58.6). The patch-similarity noise estimate is not designed for multiplicative noise. |

## 7. Statistical comparison

- **Friedman test (8 methods, paired by image):** differences between methods were highly significant for every metric at every noise level. For PSNR, SSIM and EPI, χ² ranged from 76.9 to 98.2 (p < 10⁻¹³); for ENL, p ≤ 3.8×10⁻⁴.
- **Wilcoxon signed-rank tests,** each AI model against the best traditional filter (Lee), Holm-corrected over 48 tests:
  - DnCNN > Lee for **PSNR at all three levels** (15/15 images each, adjusted p = 0.003). It was also higher for **EPI at all three levels** (adjusted p ≤ 0.008) and for SSIM at σ = 0.1 and 0.2 (adjusted p ≤ 0.013).
  - DRUNet > Lee for PSNR, SSIM and EPI at σ = 0.1 and 0.2 (adjusted p ≤ 0.003). At σ = 0.3 none of these three differences was significant.
  - Noise2Void < Lee for PSNR and SSIM at σ = 0.1 and 0.2. At σ = 0.3 it was significantly better for PSNR (adjusted p = 0.023) and EPI (adjusted p = 0.003).
  - DIP < Lee for SSIM at σ = 0.1, 0.2 and 0.3 (adjusted p ≤ 0.005). Its PSNR deficit was significant only at σ = 0.3.
- **Note on sample size:** with n = 15, the smallest possible two-sided Wilcoxon p-value is 6.1×10⁻⁵. Several comparisons hit this floor, meaning the AI model won (or lost) on every single image.

## 8. Limitations

1. **Pseudo-clean reference.** BUSI images already contain speckle. The methods are evaluated on removing *added* synthetic noise, not on recovering a speckle-free tissue image.
2. **Synthetic noise model.** Uncorrelated multiplicative Gaussian noise is simpler than real ultrasound speckle, which is spatially correlated and log-compressed. Rankings may differ on raw scanner data. This mainly affects Noise2Void, whose blind-spot assumption would be violated, and DnCNN, which learned this specific model.
3. **Reduced scale because no GPU was available.** The experiment used 128×128 images, 15 test images, 200 training images and 1,500 training iterations. Longer training would probably improve DnCNN and Noise2Void.
4. **Tuning at the edge of the grid.** Some traditional parameters were selected at the edge of their search grid, for example Lee window = 3 and Bilateral σ_color = 0.6. Slightly better settings for those filters may exist.
5. **No patient-level split.** BUSI does not provide patient identifiers, so images from the same patient may appear in both training and test sets.
6. **Zero-shot DRUNet.** DRUNet was not fine-tuned. Its σ = 0.3 weakness reflects the domain mismatch, not the architecture's capacity.

## 9. Conclusion

- **Best AI model:** DnCNN, a supervised CNN trained on the target noise model. It is significantly better than the best traditional filter in PSNR and edge preservation at every noise level, and it is the most robust method at high noise. DRUNet, used without any training, is the best method at low to moderate noise but degrades at σ = 0.3.
- **Best traditional filter:** Lee. It is the strongest classical option and remains extremely competitive given its sub-millisecond cost.
- **Trade-off:** AI methods give a consistent, statistically significant quality gain of about 1–2 dB PSNR, most importantly in edge preservation under heavy noise. The cost is training data, training time and roughly 300–1,000× more computation per image.
- **AI is not automatically better:** the self-supervised and untrained AI methods (Noise2Void, DIP) did not beat the Lee filter overall.
- **Recommendation:**
  - For quality, use DnCNN, or DRUNet when no training is possible and noise is moderate.
  - For real-time or resource-limited settings, use the Lee filter.
