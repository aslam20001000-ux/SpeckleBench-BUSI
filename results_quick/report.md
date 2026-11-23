# BUSI speckle reduction: AI vs classical — results of this run

> **QUICK MODE — smoke test only. Do not report these numbers.**

- Device: cpu (none); torch 2.15.0+cpu
- Images: 4 test / 3 val / 40 train, resized to 128x128, grayscale
- Noise: y = x·(1 + σ·n), n ~ N(0,1), σ ∈ [0.1, 0.2, 0.3]; identical noisy images for all methods
- Reference = original BUSI image (already contains native speckle). Mean ENL of the reference ROIs: 36.2

## Methods actually run

| method | category | training | train_time_s | params |
|---|---|---|---|---|
| DnCNN | AI - supervised, trained here | 40 clean BUSI imgs, synthetic speckle σ~U(0.05, 0.35), 50 iters, 64x64 patches, batch 64 | 52.900 | 556097 |
| Noise2Void | AI - self-supervised, trained here (no clean targets) | 120 noisy imgs (1 realisation per image per σ), 50 iters, 2% blind-spot pixels | 11.500 | 1927841 |
| DRUNet | AI - pretrained (Gaussian), zero-shot, NOT trained here | deepinv drunet_deepinv_gray weights; verified (+7.87 dB on Gaussian check); log-domain; only σ_in chosen on val | 0.000 | 32638656 |
| DIP | AI - untrained (no dataset), optimised per image | no training; iterations/σ chosen on 1 val imgs: {0.1: 100, 0.2: 100, 0.3: 100} | 58.800 | 1936769 |
| Median | classical filter | no training; params chosen on val: σ=0.1: {'size': 3}; σ=0.2: {'size': 3}; σ=0.3: {'size': 3} | 0.000 | 0 |
| Bilateral | classical filter | no training; params chosen on val: σ=0.1: {'sc': 0.1, 'ss': 2}; σ=0.2: {'sc': 0.3, 'ss': 2}; σ=0.3: {'sc': 0.3, 'ss': 2} | 0.000 | 0 |
| Lee | classical filter | no training; params chosen on val: σ=0.1: {'win': 5, 'cu': 0.1}; σ=0.2: {'win': 5, 'cu': 0.2}; σ=0.3: {'win': 5, 'cu': 0.4} | 0.000 | 0 |
| NLM | classical filter | no training; params chosen on val: σ=0.1: {'k': 1.5}; σ=0.2: {'k': 1.5}; σ=0.3: {'k': 1.5} | 0.000 | 0 |

## Results per noise level (mean ± std over test images)

| level | method | PSNR | SSIM | ENL | EPI | time (ms) |
|---|---|---|---|---|---|---|
| 0.100 | Noisy input | 29.549 ± 0.583 | 0.911 ± 0.013 | 25.5 ± 10.6 | 0.969 ± 0.019 | 0.0 |
| 0.100 | DnCNN | 29.034 ± 0.508 | 0.891 ± 0.019 | 27.3 ± 11.6 | 0.963 ± 0.020 | 159.5 |
| 0.100 | Noise2Void | 22.878 ± 0.924 | 0.556 ± 0.050 | 30.2 ± 25.3 | 0.796 ± 0.054 | 50.7 |
| 0.100 | DRUNet | 33.742 ± 1.020 | 0.955 ± 0.007 | 41.5 ± 24.6 | 0.977 ± 0.012 | 537.9 |
| 0.100 | DIP | 16.436 ± 0.725 | 0.249 ± 0.039 | 160.6 ± 92.8 | 0.349 ± 0.168 | 18588.7 |
| 0.100 | Median | 28.612 ± 2.008 | 0.852 ± 0.016 | 44.1 ± 23.7 | 0.945 ± 0.015 | 4.3 |
| 0.100 | Bilateral | 30.590 ± 1.033 | 0.884 ± 0.015 | 43.4 ± 22.0 | 0.958 ± 0.018 | 0.7 |
| 0.100 | Lee | 31.301 ± 0.988 | 0.931 ± 0.007 | 41.4 ± 25.7 | 0.970 ± 0.017 | 0.6 |
| 0.100 | NLM | 29.944 ± 0.486 | 0.859 ± 0.024 | 38.9 ± 18.4 | 0.950 ± 0.020 | 34.5 |
| 0.200 | Noisy input | 23.638 ± 0.608 | 0.749 ± 0.030 | 14.8 ± 3.8 | 0.896 ± 0.056 | 0.0 |
| 0.200 | DnCNN | 23.522 ± 0.571 | 0.734 ± 0.030 | 15.9 ± 4.4 | 0.891 ± 0.054 | 169.3 |
| 0.200 | Noise2Void | 21.988 ± 0.819 | 0.514 ± 0.052 | 22.8 ± 17.1 | 0.760 ± 0.068 | 46.2 |
| 0.200 | DRUNet | 29.652 ± 0.888 | 0.891 ± 0.011 | 53.0 ± 36.1 | 0.928 ± 0.040 | 510.5 |
| 0.200 | DIP | 16.392 ± 0.583 | 0.243 ± 0.036 | 171.8 ± 101.9 | 0.328 ± 0.172 | 18121.7 |
| 0.200 | Median | 26.742 ± 1.528 | 0.801 ± 0.019 | 38.1 ± 17.9 | 0.878 ± 0.042 | 4.6 |
| 0.200 | Bilateral | 26.996 ± 1.809 | 0.782 ± 0.023 | 54.0 ± 28.9 | 0.880 ± 0.031 | 0.8 |
| 0.200 | Lee | 27.523 ± 1.360 | 0.840 ± 0.017 | 48.4 ± 34.3 | 0.907 ± 0.041 | 0.6 |
| 0.200 | NLM | 25.690 ± 1.249 | 0.743 ± 0.021 | 38.5 ± 24.7 | 0.857 ± 0.052 | 38.2 |
| 0.300 | Noisy input | 20.133 ± 0.514 | 0.599 ± 0.053 | 8.5 ± 2.3 | 0.802 ± 0.079 | 0.0 |
| 0.300 | DnCNN | 20.101 ± 0.495 | 0.587 ± 0.049 | 9.0 ± 2.3 | 0.801 ± 0.075 | 163.8 |
| 0.300 | Noise2Void | 20.852 ± 0.835 | 0.462 ± 0.043 | 15.7 ± 10.6 | 0.710 ± 0.070 | 51.2 |
| 0.300 | DRUNet | 25.914 ± 0.972 | 0.782 ± 0.025 | 42.1 ± 28.4 | 0.836 ± 0.087 | 508.8 |
| 0.300 | DIP | 16.392 ± 0.575 | 0.244 ± 0.026 | 156.2 ± 79.0 | 0.341 ± 0.138 | 18618.9 |
| 0.300 | Median | 24.920 ± 1.276 | 0.741 ± 0.030 | 27.9 ± 11.8 | 0.791 ± 0.051 | 4.6 |
| 0.300 | Bilateral | 25.665 ± 1.166 | 0.756 ± 0.014 | 37.1 ± 14.9 | 0.814 ± 0.058 | 0.8 |
| 0.300 | Lee | 25.559 ± 1.944 | 0.724 ± 0.058 | 56.7 ± 36.9 | 0.792 ± 0.049 | 0.6 |
| 0.300 | NLM | 23.370 ± 1.815 | 0.653 ± 0.007 | 34.8 ± 22.2 | 0.755 ± 0.074 | 43.6 |

## Overall means (all noise levels)

| method | PSNR | SSIM | ENL | EPI | time_ms |
|---|---|---|---|---|---|
| DnCNN | 24.219 | 0.737 | 17.392 | 0.885 | 164.206 |
| Noise2Void | 21.906 | 0.511 | 22.921 | 0.755 | 49.373 |
| DRUNet | 29.769 | 0.876 | 45.534 | 0.914 | 519.105 |
| DIP | 16.407 | 0.245 | 162.852 | 0.339 | 18443.102 |
| Median | 26.758 | 0.798 | 36.721 | 0.871 | 4.480 |
| Bilateral | 27.751 | 0.807 | 44.862 | 0.884 | 0.775 |
| Lee | 28.128 | 0.832 | 48.792 | 0.890 | 0.626 |
| NLM | 26.335 | 0.751 | 37.403 | 0.854 | 38.782 |

## Ranking

Mean per-image rank across PSNR, SSIM, EPI (1 = best; ENL excluded because it rewards over-smoothing):

| method | mean_rank |
|---|---|
| DRUNet | 1.167 |
| Lee | 2.444 |
| Bilateral | 3.278 |
| Median | 4.444 |
| DnCNN | 4.806 |
| NLM | 5.000 |
| Noise2Void | 6.861 |
| DIP | 8.000 |

- **Best AI model:** DRUNet
- **Best classical filter:** Lee

- σ=0.1: DRUNet vs Lee: PSNR 33.74 vs 31.30 dB, SSIM 0.955 vs 0.931, EPI 0.977 vs 0.970, ENL 41.5 vs 41.4, time 537.9 vs 0.6 ms
- σ=0.2: DRUNet vs Lee: PSNR 29.65 vs 27.52 dB, SSIM 0.891 vs 0.840, EPI 0.928 vs 0.907, ENL 53.0 vs 48.4, time 510.5 vs 0.6 ms
- σ=0.3: DRUNet vs Lee: PSNR 25.91 vs 25.56 dB, SSIM 0.782 vs 0.724, EPI 0.836 vs 0.792, ENL 42.1 vs 56.7, time 508.8 vs 0.6 ms

## Statistics

Friedman test (all methods, paired by image):

| sigma | metric | n_images | chi2 | p_value |
|---|---|---|---|---|
| 0.100 | PSNR | 4 | 26.917 | 3.45e-04 |
| 0.100 | SSIM | 4 | 26.917 | 3.45e-04 |
| 0.100 | ENL | 4 | 18.750 | 0.009 |
| 0.100 | EPI | 4 | 27.750 | 2.44e-04 |
| 0.200 | PSNR | 4 | 26.250 | 4.54e-04 |
| 0.200 | SSIM | 4 | 27.750 | 2.44e-04 |
| 0.200 | ENL | 4 | 21.083 | 0.004 |
| 0.200 | EPI | 4 | 26.917 | 3.45e-04 |
| 0.300 | PSNR | 4 | 25.083 | 7.33e-04 |
| 0.300 | SSIM | 4 | 26.250 | 4.54e-04 |
| 0.300 | ENL | 4 | 23.083 | 0.002 |
| 0.300 | EPI | 4 | 23.917 | 0.001 |

Wilcoxon signed-rank, each AI model vs best classical (Lee); Holm-corrected over all rows. median_diff > 0 means the AI model scores higher:

| sigma | metric | comparison | median_diff | wins | p_raw | p_holm |
|---|---|---|---|---|---|---|
| 0.100 | PSNR | DnCNN vs Lee | -2.069 | 0/4 | 0.125 | 1.000 |
| 0.100 | PSNR | Noise2Void vs Lee | -8.372 | 0/4 | 0.125 | 1.000 |
| 0.100 | PSNR | DRUNet vs Lee | 2.464 | 4/4 | 0.125 | 1.000 |
| 0.100 | PSNR | DIP vs Lee | -14.351 | 0/4 | 0.125 | 1.000 |
| 0.100 | SSIM | DnCNN vs Lee | -0.038 | 0/4 | 0.125 | 1.000 |
| 0.100 | SSIM | Noise2Void vs Lee | -0.353 | 0/4 | 0.125 | 1.000 |
| 0.100 | SSIM | DRUNet vs Lee | 0.023 | 4/4 | 0.125 | 1.000 |
| 0.100 | SSIM | DIP vs Lee | -0.694 | 0/4 | 0.125 | 1.000 |
| 0.100 | ENL | DnCNN vs Lee | -7.662 | 0/4 | 0.125 | 1.000 |
| 0.100 | ENL | Noise2Void vs Lee | -8.034 | 1/4 | 0.250 | 1.000 |
| 0.100 | ENL | DRUNet vs Lee | 0.504 | 3/4 | 0.875 | 1.000 |
| 0.100 | ENL | DIP vs Lee | 109.194 | 4/4 | 0.125 | 1.000 |
| 0.100 | EPI | DnCNN vs Lee | -0.006 | 0/4 | 0.125 | 1.000 |
| 0.100 | EPI | Noise2Void vs Lee | -0.162 | 0/4 | 0.125 | 1.000 |
| 0.100 | EPI | DRUNet vs Lee | 0.005 | 4/4 | 0.125 | 1.000 |
| 0.100 | EPI | DIP vs Lee | -0.673 | 0/4 | 0.125 | 1.000 |
| 0.200 | PSNR | DnCNN vs Lee | -3.577 | 0/4 | 0.125 | 1.000 |
| 0.200 | PSNR | Noise2Void vs Lee | -5.572 | 0/4 | 0.125 | 1.000 |
| 0.200 | PSNR | DRUNet vs Lee | 2.361 | 4/4 | 0.125 | 1.000 |
| 0.200 | PSNR | DIP vs Lee | -10.585 | 0/4 | 0.125 | 1.000 |
| 0.200 | SSIM | DnCNN vs Lee | -0.106 | 0/4 | 0.125 | 1.000 |
| 0.200 | SSIM | Noise2Void vs Lee | -0.300 | 0/4 | 0.125 | 1.000 |
| 0.200 | SSIM | DRUNet vs Lee | 0.055 | 4/4 | 0.125 | 1.000 |
| 0.200 | SSIM | DIP vs Lee | -0.608 | 0/4 | 0.125 | 1.000 |
| 0.200 | ENL | DnCNN vs Lee | -19.830 | 0/4 | 0.125 | 1.000 |
| 0.200 | ENL | Noise2Void vs Lee | -23.722 | 0/4 | 0.125 | 1.000 |
| 0.200 | ENL | DRUNet vs Lee | 3.411 | 4/4 | 0.125 | 1.000 |
| 0.200 | ENL | DIP vs Lee | 117.642 | 4/4 | 0.125 | 1.000 |
| 0.200 | EPI | DnCNN vs Lee | -0.011 | 0/4 | 0.125 | 1.000 |
| 0.200 | EPI | Noise2Void vs Lee | -0.144 | 0/4 | 0.125 | 1.000 |
| 0.200 | EPI | DRUNet vs Lee | 0.021 | 4/4 | 0.125 | 1.000 |
| 0.200 | EPI | DIP vs Lee | -0.640 | 0/4 | 0.125 | 1.000 |
| 0.300 | PSNR | DnCNN vs Lee | -4.951 | 0/4 | 0.125 | 1.000 |
| 0.300 | PSNR | Noise2Void vs Lee | -4.528 | 0/4 | 0.125 | 1.000 |
| 0.300 | PSNR | DRUNet vs Lee | 0.789 | 3/4 | 0.875 | 1.000 |
| 0.300 | PSNR | DIP vs Lee | -8.384 | 0/4 | 0.125 | 1.000 |
| 0.300 | SSIM | DnCNN vs Lee | -0.148 | 0/4 | 0.125 | 1.000 |
| 0.300 | SSIM | Noise2Void vs Lee | -0.239 | 0/4 | 0.125 | 1.000 |
| 0.300 | SSIM | DRUNet vs Lee | 0.061 | 3/4 | 0.250 | 1.000 |
| 0.300 | SSIM | DIP vs Lee | -0.466 | 0/4 | 0.125 | 1.000 |
| 0.300 | ENL | DnCNN vs Lee | -34.072 | 0/4 | 0.125 | 1.000 |
| 0.300 | ENL | Noise2Void vs Lee | -31.899 | 0/4 | 0.125 | 1.000 |
| 0.300 | ENL | DRUNet vs Lee | -12.577 | 0/4 | 0.125 | 1.000 |
| 0.300 | ENL | DIP vs Lee | 94.347 | 4/4 | 0.125 | 1.000 |
| 0.300 | EPI | DnCNN vs Lee | 0.011 | 2/4 | 0.875 | 1.000 |
| 0.300 | EPI | Noise2Void vs Lee | -0.089 | 0/4 | 0.125 | 1.000 |
| 0.300 | EPI | DRUNet vs Lee | 0.056 | 3/4 | 0.250 | 1.000 |
| 0.300 | EPI | DIP vs Lee | -0.503 | 0/4 | 0.125 | 1.000 |
