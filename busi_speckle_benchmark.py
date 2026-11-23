"""
BUSI speckle-reduction benchmark: 4 AI models vs 4 classical filters.

AI models (each in a different training regime):
  1. DnCNN        - supervised, trained from scratch here on the BUSI train split
  2. Noise2Void   - self-supervised (blind-spot U-Net), trained here on NOISY images only
  3. DRUNet       - pretrained (deepinv grayscale weights, Gaussian denoiser), zero-shot,
                    applied in the log domain (homomorphic transform); no training here
  4. Deep Image Prior - untrained network, optimised per image (no dataset training)

Classical filters: Median, Bilateral, Lee, Non-Local Means.

Every method is run on the SAME resized test images with the SAME pre-generated
noisy realisations. Hyper-parameters of classical filters, the DRUNet noise level and
the DIP iteration count are selected on a separate validation split (never on test).

Metrics: PSNR, SSIM, ENL (homogeneous ROI), EPI (edge preservation index), time.

Usage (Colab):
  !python busi_speckle_benchmark.py --quick            # smoke test, ~3 min
  !python busi_speckle_benchmark.py                    # full experiment
  !python busi_speckle_benchmark.py --data_dir /content/Dataset_BUSI_with_GT
"""
import argparse
import copy
import glob
import json
import os
import random
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage, stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_nl_means, estimate_sigma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None,
                   help="Folder containing benign/ malignant/ normal/ (BUSI). "
                        "If omitted, downloads via kagglehub.")
    p.add_argument("--kaggle_handle", default="aryashah2k/breast-ultrasound-images-dataset")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--n_val", type=int, default=10)
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--noise_levels", type=float, nargs="+", default=[0.1, 0.2, 0.3],
                   help="std of multiplicative speckle: y = x * (1 + sigma * n), n ~ N(0,1)")
    p.add_argument("--dncnn_iters", type=int, default=4000)
    p.add_argument("--n2v_iters", type=int, default=4000)
    p.add_argument("--dip_max_iters", type=int, default=2000)
    p.add_argument("--dip_val_images", type=int, default=3)
    p.add_argument("--cpu_timing_images", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip", nargs="*", default=[],
                   help="methods to skip, e.g. --skip DIP DRUNet")
    p.add_argument("--train_batch", type=int, default=None, help="default: 64 (DnCNN) / 32 (N2V)")
    p.add_argument("--patch", type=int, default=64, help="training patch size (multiple of 8)")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--cpu_preset", action="store_true",
                   help="lighter settings for a laptop CPU without NVIDIA GPU (same for ALL methods)")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke test to check the pipeline; NOT for reporting")
    a = p.parse_args()
    if a.cpu_preset:
        a.img_size, a.n_test, a.n_train = 128, 15, 200
        a.dncnn_iters, a.n2v_iters, a.dip_max_iters, a.dip_val_images = 1500, 1500, 1000, 2
        a.train_batch, a.patch = 16, 48
    if a.quick:
        a.img_size, a.n_test, a.n_val, a.n_train = 128, 4, 3, 40
        a.dncnn_iters, a.n2v_iters, a.dip_max_iters = 50, 50, 100
        a.dip_val_images, a.cpu_timing_images = 1, 2
        a.train_batch, a.patch = 8, 48
    return a


AI_METHODS = ["DnCNN", "Noise2Void", "DRUNet", "DIP"]
CLASSICAL_METHODS = ["Median", "Bilateral", "Lee", "NLM"]
ALL_METHODS = AI_METHODS + CLASSICAL_METHODS
QUALITY_METRICS = ["PSNR", "SSIM", "ENL", "EPI"]
RANK_METRICS = ["PSNR", "SSIM", "EPI"]   # fidelity metrics used to pick "best" (ENL rewards blur)

# Fixed categorical palette (validated 8-hue order); colour follows the method, never its rank.
COLORS = dict(zip(ALL_METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                "#e87ba4", "#008300", "#6250d6", "#e34948"]))
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def log(*a):
    print(*a, flush=True)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
def find_busi(data_dir, handle):
    if data_dir is None:
        import kagglehub
        log(f"Downloading BUSI with kagglehub: {handle}")
        data_dir = kagglehub.dataset_download(handle)
    files, seen = [], set()
    for cls in ["benign", "malignant", "normal"]:
        hits = glob.glob(os.path.join(data_dir, "**", cls, "*.png"), recursive=True)
        for h in sorted(hits):
            b = os.path.basename(h)
            if "_mask" in b or (cls, b) in seen:       # drop masks and duplicate copies
                continue
            seen.add((cls, b))
            files.append((h, cls))
    if not files:
        raise FileNotFoundError(f"No BUSI images (benign/malignant/normal *.png) under {data_dir}")
    log(f"Found {len(files)} BUSI images in {data_dir}: " +
        ", ".join(f"{c}={sum(1 for _, k in files if k == c)}" for c in ["benign", "malignant", "normal"]))
    return files


def split_files(files, n_test, n_val, n_train, seed):
    """Stratified test split; val/train drawn from the rest. Splits are disjoint."""
    rng = np.random.RandomState(seed)
    by = OrderedDict()
    for f, c in files:
        by.setdefault(c, []).append(f)
    test, rest = [], []
    for c, lst in by.items():
        lst = list(lst)
        rng.shuffle(lst)
        k = max(1, int(round(n_test * len(lst) / len(files))))
        test += [(f, c) for f in lst[:k]]
        rest += [(f, c) for f in lst[k:]]
    test = test[:n_test]
    rng.shuffle(rest)
    return rest[n_val:n_val + n_train], rest[:n_val], test


def load_img(path, size):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(path)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return im.astype(np.float32) / 255.0


def add_speckle(x, sigma, rng):
    n = rng.standard_normal(x.shape).astype(np.float32)
    return np.clip(x * (1.0 + sigma * n), 0, 1).astype(np.float32)


def add_speckle_torch(x, sigma):
    return (x * (1.0 + sigma * torch.randn_like(x))).clamp(0, 1)


# ----------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------
def m_psnr(ref, x):
    return float(peak_signal_noise_ratio(ref, np.clip(x, 0, 1).astype(np.float32), data_range=1.0))


def m_ssim(ref, x):
    return float(structural_similarity(ref, np.clip(x, 0, 1).astype(np.float32), data_range=1.0))


def homogeneous_roi(ref, win=24, stride=8, min_mean=0.15):
    """Most homogeneous (lowest CV) window of the reference image with mean > min_mean.
    Chosen once per image from the reference and reused for every method."""
    H, W = ref.shape
    best, best_cv, brightest, best_mean = None, np.inf, (0, 0), -1
    for i in range(0, H - win + 1, stride):
        for j in range(0, W - win + 1, stride):
            p = ref[i:i + win, j:j + win]
            m = float(p.mean())
            if m > best_mean:
                best_mean, brightest = m, (i, j)
            if m < min_mean:
                continue
            cv = float(p.std()) / m
            if cv < best_cv:
                best_cv, best = cv, (i, j)
    i, j = best if best is not None else brightest
    return (i, j, win)


def m_enl(x, roi):
    i, j, w = roi
    p = np.clip(x, 0, 1)[i:i + w, j:j + w].astype(np.float64)
    return float(p.mean() ** 2 / max(p.var(), 1e-12))


def m_epi(ref, x, sigma=1.5):
    """Edge Preservation Index (Sattar et al. 1997) using a LoG high-pass at sigma=1.5
    so it measures structural edges rather than pixel-level speckle."""
    a = ndimage.gaussian_laplace(ref.astype(np.float64), sigma)
    b = ndimage.gaussian_laplace(np.clip(x, 0, 1).astype(np.float64), sigma)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


# ----------------------------------------------------------------------------------
# Classical filters (+ grids searched on the validation split)
# ----------------------------------------------------------------------------------
def f_median(y, size):
    return ndimage.median_filter(y, size=size, mode="reflect")


def f_bilateral(y, sc, ss):
    return cv2.bilateralFilter(y.astype(np.float32), d=0, sigmaColor=sc, sigmaSpace=ss)


def f_lee(y, win, cu):
    m = ndimage.uniform_filter(y, win)
    m2 = ndimage.uniform_filter(y * y, win)
    var = np.maximum(m2 - m * m, 0)
    cy2 = var / np.maximum(m * m, 1e-8)
    w = np.clip(1.0 - (cu * cu) / np.maximum(cy2, 1e-8), 0, 1)
    return (m + w * (y - m)).astype(np.float32)


def f_nlm(y, k):
    s = max(float(estimate_sigma(y, channel_axis=None)), 1e-3)
    return denoise_nl_means(y, h=k * s, sigma=s, patch_size=5, patch_distance=6,
                            fast_mode=True, channel_axis=None).astype(np.float32)


CLASSICAL = OrderedDict([
    ("Median", (f_median, [dict(size=s) for s in (3, 5, 7, 9)])),
    ("Bilateral", (f_bilateral, [dict(sc=sc, ss=ss) for sc in (0.05, 0.1, 0.2, 0.3, 0.4, 0.6)
                                 for ss in (1, 2, 4, 6)])),
    ("Lee", (f_lee, [dict(win=w, cu=c) for w in (3, 5, 7, 9) for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)])),
    ("NLM", (f_nlm, [dict(k=k) for k in (0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)])),
])


def tune_classical(val_clean, val_noisy, levels, skip):
    tuned = {}
    for name, (f, grid) in CLASSICAL.items():
        if name in skip:
            continue
        tuned[name] = {}
        for li, s in enumerate(levels):
            scores = [np.mean([m_psnr(x, f(y, **cfg)) for x, y in zip(val_clean, val_noisy[li])])
                      for cfg in grid]
            b = int(np.argmax(scores))
            tuned[name][li] = grid[b]
            log(f"  [tune] {name:9s} sigma={s}: best {grid[b]} (val PSNR {scores[b]:.2f})")
    return tuned


# ----------------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------------
class DnCNN(nn.Module):
    """Zhang et al. 2017. 17 conv layers, residual learning (predicts the noise)."""

    def __init__(self, depth=17, ch=64):
        super().__init__()
        layers = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, y):
        return y - self.body(y)


def conv_block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1, inplace=True))


class UNet(nn.Module):
    """Small U-Net (3 down-samplings). Used by Noise2Void and Deep Image Prior."""

    def __init__(self, in_ch=1, out_ch=1, base=32, depth=3):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.downs.append(conv_block(c, ch))
            c = ch
        self.mid = conv_block(chs[-2], chs[-1])
        self.ups, self.decs = nn.ModuleList(), nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.decs.append(conv_block(ch * 2, ch))
        self.out = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        skips, h = [], x
        for d in self.downs:
            h = d(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        h = self.mid(h)
        for up, dec, s in zip(self.ups, self.decs, reversed(skips)):
            h = dec(torch.cat([up(h), s], 1))
        return self.out(h)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def sample_patches(imgs, bs, ps):
    N, _, H, W = imgs.shape
    idx = torch.randint(0, N, (bs,)).tolist()
    ii = torch.randint(0, H - ps + 1, (bs,)).tolist()
    jj = torch.randint(0, W - ps + 1, (bs,)).tolist()
    out = torch.stack([imgs[a, :, b:b + ps, c:c + ps] for a, b, c in zip(idx, ii, jj)])
    if random.random() < 0.5:
        out = out.flip(-1)
    if random.random() < 0.5:
        out = out.flip(-2)
    return out


def train_dncnn(train_t, iters, dev, sig_range, bs=64, ps=64):
    net = DnCNN().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.6 * iters), int(0.85 * iters)], 0.1)
    losses = []
    net.train()
    for t in range(iters):
        x = sample_patches(train_t, bs, ps)
        s = torch.empty(x.size(0), 1, 1, 1, device=dev).uniform_(*sig_range)
        y = add_speckle_torch(x, s)
        loss = F.mse_loss(net(y), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (t + 1) % max(1, iters // 10) == 0:
            log(f"  [DnCNN] iter {t + 1}/{iters} loss {np.mean(losses[-100:]):.5f}")
    return net.eval(), losses


def n2v_mask(y, p=0.02, r=2):
    """Blind-spot input: replace a random p fraction of pixels by a random neighbour
    (within +-r, never itself). Returns masked input and the boolean mask."""
    B, _, H, W = y.shape
    dev = y.device
    ii = torch.arange(H, device=dev).view(1, H, 1).expand(B, H, W)
    jj = torch.arange(W, device=dev).view(1, 1, W).expand(B, H, W)
    dy = torch.randint(-r, r + 1, (B, H, W), device=dev)
    dx = torch.randint(-r, r + 1, (B, H, W), device=dev)
    dx = torch.where((dx == 0) & (dy == 0), torch.ones_like(dx), dx)
    ni = (ii + dy).clamp(0, H - 1)
    nj = (jj + dx).clamp(0, W - 1)
    bi = torch.arange(B, device=dev).view(B, 1, 1).expand(B, H, W)
    neigh = y[bi, 0, ni, nj]
    m = torch.rand(B, H, W, device=dev) < p
    inp = torch.where(m, neigh, y[:, 0]).unsqueeze(1)
    return inp, m


def train_n2v(noisy_t, iters, dev, bs=32, ps=64):
    net = UNet(1, 1, base=32, depth=3).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=4e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    losses = []
    net.train()
    for t in range(iters):
        y = sample_patches(noisy_t, bs, ps)
        inp, m = n2v_mask(y)
        out = net(inp)
        mf = m.float()
        loss = (((out[:, 0] - y[:, 0]) ** 2) * mf).sum() / mf.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (t + 1) % max(1, iters // 10) == 0:
            log(f"  [N2V] iter {t + 1}/{iters} masked loss {np.mean(losses[-100:]):.5f}")
    return net.eval(), losses


@torch.no_grad()
def run_net(net, y, dev):
    t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None, None].to(dev)
    return net(t)[0, 0].clamp(0, 1).cpu().numpy()


# ---- DRUNet (pretrained, deepinv) in the log domain ----
LOG_EPS = 1e-2
LMIN, LMAX = float(np.log(LOG_EPS)), float(np.log(1 + LOG_EPS))


def load_and_verify_drunet(dev, check_imgs):
    """Downloads deepinv grayscale DRUNet weights and verifies them at runtime on a
    Gaussian-denoising sanity check (the task they were trained for)."""
    import deepinv as dinv
    net = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained="download", device=dev).eval()
    rng = np.random.default_rng(123)
    gains = []
    for x in check_imgs:
        y = np.clip(x + 0.1 * rng.standard_normal(x.shape).astype(np.float32), 0, 1).astype(np.float32)
        with torch.no_grad():
            out = net(torch.from_numpy(y)[None, None].to(dev), 0.1)[0, 0].clamp(0, 1).cpu().numpy()
        gains.append(m_psnr(x, out) - m_psnr(x, y))
    g = float(np.mean(gains))
    log(f"  [DRUNet] weight check: Gaussian sigma=0.1 PSNR gain = {g:+.2f} dB")
    if g < 2.0:
        raise RuntimeError(f"DRUNet weight verification failed (gain {g:.2f} dB < 2 dB)")
    return net, g


@torch.no_grad()
def drunet_speckle(net, y, sig, dev):
    z = (np.log(y + LOG_EPS) - LMIN) / (LMAX - LMIN)
    t = torch.from_numpy(z.astype(np.float32))[None, None].to(dev)
    o = net(t, float(sig))[0, 0].cpu().numpy().astype(np.float64)
    o = o * (LMAX - LMIN) + LMIN
    return np.clip(np.exp(o) - LOG_EPS, 0, 1).astype(np.float32)


DRUNET_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def tune_drunet(net, val_clean, val_noisy, levels, dev):
    chosen = {}
    for li, s in enumerate(levels):
        cands = [f * s / (LMAX - LMIN) for f in DRUNET_FACTORS]
        scores = [np.mean([m_psnr(x, drunet_speckle(net, y, c, dev)) for x, y in zip(val_clean, val_noisy[li])])
                  for c in cands]
        b = int(np.argmax(scores))
        chosen[li] = cands[b]
        log(f"  [tune] DRUNet    sigma={s}: log-domain sigma_in={cands[b]:.4f} (val PSNR {scores[b]:.2f})")
    return chosen


# ---- Deep Image Prior ----
def dip_denoise(y, n_iter, dev, seed=0, ref=None, record_every=0):
    torch.manual_seed(seed)
    net = UNet(in_ch=32, out_ch=1, base=32, depth=3).to(dev)
    H, W = y.shape
    z = torch.rand(1, 32, H, W, device=dev) * 0.1
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None, None].to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    avg, hist = None, []
    for it in range(1, n_iter + 1):
        out = torch.sigmoid(net(z + torch.randn_like(z) / 30.0))
        loss = F.mse_loss(out, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        o = out.detach()
        avg = o if avg is None else 0.99 * avg + 0.01 * o     # exponential average (Ulyanov et al.)
        if record_every and it % record_every == 0 and ref is not None:
            hist.append((it, m_psnr(ref, avg[0, 0].cpu().numpy())))
    return avg[0, 0].clamp(0, 1).cpu().numpy(), hist


def select_dip_iters(val_clean, val_noisy, levels, max_iters, dev, n_imgs):
    chosen, curves = {}, {}
    step = max(10, max_iters // 20)
    for li, s in enumerate(levels):
        hs = []
        for k in range(min(n_imgs, len(val_clean))):
            _, h = dip_denoise(val_noisy[li][k], max_iters, dev, seed=k, ref=val_clean[k], record_every=step)
            hs.append([p for _, p in h])
        its = [i for i, _ in h]
        mean_curve = np.mean(hs, axis=0)
        chosen[li] = int(its[int(np.argmax(mean_curve))])
        curves[s] = (its, mean_curve.tolist())
        log(f"  [tune] DIP       sigma={s}: {chosen[li]} iterations (val PSNR {mean_curve.max():.2f})")
    return chosen, curves


# ----------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------
def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, running = len(idx), 0.0
    for r, i in enumerate(idx):
        running = max(running, min(1.0, (m - r) * p[i]))
        out[i] = running
    return out


def mean_ranks(df, methods):
    d = df[df.method.isin(methods)].copy()
    rk = [d.groupby(["level", "image"])[m].rank(ascending=False) for m in RANK_METRICS]
    d["rank"] = np.mean([r.values for r in rk], axis=0)
    return d.groupby("method")["rank"].mean().sort_values()


def stat_tests(df, methods, ai, best_trad, levels):
    fr_rows, w_rows = [], []
    for s in levels:
        d = df[df.level == s]
        for met in QUALITY_METRICS:
            piv = d.pivot(index="image", columns="method", values=met)[methods].dropna()
            if len(methods) >= 3 and len(piv) >= 3:
                fr = stats.friedmanchisquare(*[piv[c].values for c in methods])
                fr_rows.append(dict(sigma=s, metric=met, n_images=len(piv),
                                    chi2=fr.statistic, p_value=fr.pvalue))
            for a in ai:
                if a == best_trad:
                    continue
                x, y = piv[a].values, piv[best_trad].values
                try:
                    p = stats.wilcoxon(x, y).pvalue
                except ValueError:
                    p = np.nan
                w_rows.append(dict(sigma=s, metric=met, comparison=f"{a} vs {best_trad}",
                                   median_diff=float(np.median(x - y)),
                                   wins=f"{int((x > y).sum())}/{len(x)}", p_raw=p))
    fr_df, w_df = pd.DataFrame(fr_rows), pd.DataFrame(w_rows)
    if len(w_df):
        w_df["p_holm"] = holm(w_df["p_raw"].values)
    return fr_df, w_df


def md_table(df, fmt=".3f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if np.isnan(v) else (f"{v:.2e}" if 0 < abs(v) < 1e-3 else format(v, fmt)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------
def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def ci95(v):
    v = np.asarray(v, float)
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def plot_metric_bars(df, methods, levels, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), facecolor=SURFACE)
    n = len(methods)
    w = 0.8 / n
    for ax, met in zip(axes.ravel(), QUALITY_METRICS):
        style_ax(ax)
        for k, m in enumerate(methods):
            vals = [df[(df.method == m) & (df.level == s)][met] for s in levels]
            xs = np.arange(len(levels)) - 0.4 + w * (k + 0.5)
            ax.bar(xs, [v.mean() for v in vals], width=w, color=COLORS[m],
                   edgecolor=SURFACE, linewidth=0.8, hatch="//" if m in CLASSICAL_METHODS else None,
                   label=m + (" (classical)" if m in CLASSICAL_METHODS else " (AI)"))
            ax.errorbar(xs, [v.mean() for v in vals], yerr=[ci95(v) for v in vals],
                        fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0)
        base = [df[(df.method == "Noisy input") & (df.level == s)][met].mean() for s in levels]
        lo = min(min(base), df[df.method.isin(methods)].groupby(["method", "level"])[met].mean().min())
        for x0, b in zip(range(len(levels)), base):
            ax.hlines(b, x0 - 0.42, x0 + 0.42, colors=MUTED, linestyles="--", linewidth=1)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"σ = {s}" for s in levels])
        ax.set_title(met + (" (log scale)" if met == "ENL" else "") + "   — dashed = noisy input",
                     color=INK, fontsize=11, loc="left")
        if met == "ENL":
            ax.set_yscale("log")
        elif met in ("SSIM", "EPI"):
            ax.set_ylim(max(0, lo - 0.05), 1.0)
        else:
            ax.set_ylim(lo - 2, None)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Mean over test images, error bars = 95% CI", y=0.995, color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_vs_noise(df, methods, levels, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    for ax, met in zip(axes, ["PSNR", "SSIM"]):
        style_ax(ax)
        for m in methods + ["Noisy input"]:
            ys = [df[(df.method == m) & (df.level == s)][met].mean() for s in levels]
            if m == "Noisy input":
                ax.plot(levels, ys, "--", color=MUTED, linewidth=1.5, label=m)
            else:
                ax.plot(levels, ys, "-" if m in AI_METHODS else ":", color=COLORS[m], linewidth=2,
                        marker="o" if m in AI_METHODS else "s", markersize=6, label=m)
        ax.set_xticks(levels)
        ax.set_xlabel("speckle σ", color=INK2)
        ax.set_title(f"{met} vs noise level (solid = AI, dotted = classical)", color=INK, fontsize=11, loc="left")
    axes[1].legend(frameon=False, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_boxes(df, methods, levels, met, path):
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4.5), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, levels):
        style_ax(ax)
        data = [df[(df.method == m) & (df.level == s)][met].values for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color=INK, linewidth=1.2),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED, markeredgecolor="none"))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[m])
            patch.set_edgecolor(SURFACE)
            if m in CLASSICAL_METHODS:
                patch.set_hatch("//")
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_title(f"{met} per image, σ = {s}", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_time(summary_time, path, size=256):
    d = summary_time.sort_values("ms_per_image")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(d) + 1.5), facecolor=SURFACE)
    style_ax(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.barh(range(len(d)), d.ms_per_image, color=[COLORS[m] for m in d.method], height=0.6,
            hatch=None, edgecolor=SURFACE)
    for k, (m, v, dv) in enumerate(zip(d.method, d.ms_per_image, d.device)):
        ax.text(v * 1.08, k, f"{v:,.1f} ms ({dv})", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.method)
    ax.set_xscale("log")
    ax.set_xlabel(f"mean processing time per {size}×{size} image (log scale)", color=INK2)
    ax.set_title("Inference / processing time (excludes one-off training)", color=INK, fontsize=11, loc="left")
    ax.set_xlim(None, d.ms_per_image.max() * 8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_examples(examples, methods, levels, li, path):
    keys = [k for k in examples if k[0] == li]
    cols = ["Reference", "Noisy input"] + methods
    fig, axes = plt.subplots(len(keys), len(cols), figsize=(2.1 * len(cols), 2.9 * len(keys) + 0.4), facecolor="white")
    axes = np.atleast_2d(axes)
    for r, k in enumerate(keys):
        ex = examples[k]
        for c, name in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(ex[name], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            t = name if name == "Reference" else f"{name}\n{m_psnr(ex['Reference'], ex[name]):.2f} dB / " \
                                                 f"{m_ssim(ex['Reference'], ex[name]):.3f}"
            ax.set_title(t, fontsize=8, color=INK)
        axes[r, 0].set_ylabel(ex["label"], fontsize=8)
    fig.suptitle(f"Visual comparison at σ = {levels[li]}  (PSNR / SSIM vs reference)", fontsize=10)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training(curves, dip_curves, path):
    n = len(curves) + (1 if dip_curves else 0)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (name, losses) in zip(axes, curves.items()):
        style_ax(ax)
        l = np.asarray(losses)
        k = max(1, len(l) // 100)
        sm = np.convolve(l, np.ones(k) / k, mode="valid")
        ax.plot(sm, color=COLORS[name], linewidth=2)
        ax.set_yscale("log")
        ax.set_xlabel("iteration", color=INK2)
        ax.set_title(f"{name} training loss (moving avg)", color=INK, fontsize=11, loc="left")
    if dip_curves:
        ax = axes[-1]
        style_ax(ax)
        shades = ["#86b6ef", "#2a78d6", "#184f95"]
        for k, (s, (its, ps)) in enumerate(dip_curves.items()):
            ax.plot(its, ps, color=shades[k % 3], linewidth=2, label=f"σ = {s}")
        ax.set_xlabel("DIP iteration", color=INK2)
        ax.set_title("DIP validation PSNR (iteration selection)", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------------
def main():
    # Windows consoles/pipes default to cp1252, which cannot print "σ" and would crash mid-run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    out, fig_dir = args.out_dir, os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    seed_all(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "none"
    log(f"Device: {dev} ({gpu}) | torch {torch.__version__}")
    if args.quick:
        log("QUICK MODE: smoke test only, numbers are NOT meaningful.")

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # ---------- data ----------
    files = find_busi(args.data_dir, args.kaggle_handle)
    train_f, val_f, test_f = split_files(files, args.n_test, args.n_val, args.n_train, args.seed)
    log(f"Split: train={len(train_f)} val={len(val_f)} test={len(test_f)} (disjoint)")
    with open(os.path.join(out, "split.json"), "w") as fh:
        json.dump(dict(train=train_f, val=val_f, test=test_f), fh, indent=1)
    S, levels = args.img_size, list(args.noise_levels)
    train_clean = [load_img(f, S) for f, _ in train_f]
    val_clean = [load_img(f, S) for f, _ in val_f]
    test_clean = [load_img(f, S) for f, _ in test_f]
    test_labels = [f"{c}: {os.path.basename(f)}" for f, c in test_f]
    # identical, pre-generated noisy realisations shared by every method
    val_noisy = {li: [add_speckle(x, s, np.random.default_rng([args.seed, 1, li, i]))
                      for i, x in enumerate(val_clean)] for li, s in enumerate(levels)}
    test_noisy = {li: [add_speckle(x, s, np.random.default_rng([args.seed, 2, li, i]))
                       for i, x in enumerate(test_clean)] for li, s in enumerate(levels)}
    rois = [homogeneous_roi(x) for x in test_clean]
    sig_range = (max(0.0, min(levels) - 0.05), max(levels) + 0.05)
    skip = set(args.skip)

    info = OrderedDict()   # method -> description row
    methods = OrderedDict()  # method -> (callable(y, li), device str)
    train_curves, dip_curves = {}, {}

    # ---------- AI 1: DnCNN (supervised) ----------
    if "DnCNN" not in skip:
        log("\n== DnCNN: supervised training on BUSI train split ==")
        train_t = torch.from_numpy(np.stack(train_clean)[:, None]).to(dev)
        t0 = time.time()
        dncnn, losses = train_dncnn(train_t, args.dncnn_iters, dev, sig_range,
                                    bs=args.train_batch or 64, ps=args.patch)
        sync()
        tt = time.time() - t0
        train_curves["DnCNN"] = losses
        methods["DnCNN"] = (lambda y, li, n=dncnn: run_net(n, y, dev), dev.type)
        info["DnCNN"] = dict(category="AI - supervised, trained here",
                             training=f"{len(train_clean)} clean BUSI imgs, synthetic speckle σ~U{sig_range}, "
                                      f"{args.dncnn_iters} iters, {args.patch}x{args.patch} patches, batch {args.train_batch or 64}",
                             train_time_s=round(tt, 1), params=n_params(dncnn))
        del train_t

    # ---------- AI 2: Noise2Void (self-supervised) ----------
    if "Noise2Void" not in skip:
        log("\n== Noise2Void: self-supervised training on NOISY train images only ==")
        noisy_train = [add_speckle(x, s, np.random.default_rng([args.seed, 0, li, i]))
                       for li, s in enumerate(levels) for i, x in enumerate(train_clean)]
        noisy_t = torch.from_numpy(np.stack(noisy_train)[:, None]).to(dev)
        t0 = time.time()
        n2v, losses = train_n2v(noisy_t, args.n2v_iters, dev, bs=args.train_batch or 32, ps=args.patch)
        sync()
        tt = time.time() - t0
        train_curves["Noise2Void"] = losses
        methods["Noise2Void"] = (lambda y, li, n=n2v: run_net(n, y, dev), dev.type)
        info["Noise2Void"] = dict(category="AI - self-supervised, trained here (no clean targets)",
                                  training=f"{len(noisy_train)} noisy imgs (1 realisation per image per σ), "
                                           f"{args.n2v_iters} iters, 2% blind-spot pixels",
                                  train_time_s=round(tt, 1), params=n_params(n2v))
        del noisy_t

    # ---------- AI 3: DRUNet (pretrained, zero-shot) ----------
    if "DRUNet" not in skip:
        log("\n== DRUNet: pretrained deepinv weights, no training ==")
        try:
            drunet, gain = load_and_verify_drunet(dev, val_clean[:3])
            dr_sig = tune_drunet(drunet, val_clean, val_noisy, levels, dev)
            methods["DRUNet"] = (lambda y, li, n=drunet: drunet_speckle(n, y, dr_sig[li], dev), dev.type)
            info["DRUNet"] = dict(category="AI - pretrained (Gaussian), zero-shot, NOT trained here",
                                  training=f"deepinv drunet_deepinv_gray weights; verified (+{gain:.2f} dB on "
                                           f"Gaussian check); log-domain; only σ_in chosen on val",
                                  train_time_s=0.0, params=n_params(drunet))
        except Exception as e:  # noqa: BLE001
            log(f"  !! DRUNet NOT RUN: {e!r}")
            info["DRUNet"] = dict(category="NOT RUN", training=f"failed: {e!r}", train_time_s=np.nan, params=0)

    # ---------- AI 4: Deep Image Prior (untrained) ----------
    if "DIP" not in skip:
        log("\n== Deep Image Prior: untrained, per-image optimisation ==")
        t0 = time.time()
        dip_iters, dip_curves = select_dip_iters(val_clean, val_noisy, levels, args.dip_max_iters, dev,
                                                 args.dip_val_images)
        sel_t = time.time() - t0
        methods["DIP"] = (lambda y, li: dip_denoise(y, dip_iters[li], dev, seed=0)[0], dev.type)
        info["DIP"] = dict(category="AI - untrained (no dataset), optimised per image",
                           training=f"no training; iterations/σ chosen on {args.dip_val_images} val imgs: "
                                    f"{ {levels[k]: v for k, v in dip_iters.items()} }",
                           train_time_s=round(sel_t, 1),
                           params=n_params(UNet(in_ch=32, out_ch=1, base=32, depth=3)))

    # ---------- classical ----------
    log("\n== Classical filters: parameter selection on validation split ==")
    tuned = tune_classical(val_clean, val_noisy, levels, skip)
    for name in CLASSICAL_METHODS:
        if name in tuned:
            f = CLASSICAL[name][0]
            methods[name] = (lambda y, li, f=f, name=name: f(y, **tuned[name][li]), "cpu")
            info[name] = dict(category="classical filter",
                              training="no training; params chosen on val: " +
                                       "; ".join(f"σ={levels[li]}: {c}" for li, c in tuned[name].items()),
                              train_time_s=0.0, params=0)

    order = [m for m in ALL_METHODS if m in methods]
    ai_run = [m for m in AI_METHODS if m in methods]
    cl_run = [m for m in CLASSICAL_METHODS if m in methods]

    # ---------- evaluation ----------
    log("\n== Evaluation on the test split ==")
    for m in order:   # warm-up (CUDA kernels / cudnn autotune) so timing is not biased
        if m != "DIP":
            methods[m][0](test_noisy[0][0], 0)
    ex_idx = sorted({next(i for i, (_, c) in enumerate(test_f) if c == cls)
                     for cls in ("benign", "malignant") if any(c == cls for _, c in test_f)})
    examples, rows = {}, []
    for li, s in enumerate(levels):
        for i, (x, y) in enumerate(zip(test_clean, test_noisy[li])):
            base = dict(level=s, image=i, file=test_labels[i])
            rows.append(dict(base, method="Noisy input", PSNR=m_psnr(x, y), SSIM=m_ssim(x, y),
                             ENL=m_enl(y, rois[i]), EPI=m_epi(x, y), time_ms=0.0))
            if i in ex_idx:
                examples[(li, i)] = {"Reference": x, "Noisy input": y, "label": test_labels[i]}
            for m in order:
                fn, _ = methods[m]
                sync()
                t0 = time.perf_counter()
                o = fn(y, li)
                sync()
                dt = (time.perf_counter() - t0) * 1000
                o = np.clip(o, 0, 1).astype(np.float32)
                rows.append(dict(base, method=m, PSNR=m_psnr(x, o), SSIM=m_ssim(x, o),
                                 ENL=m_enl(o, rois[i]), EPI=m_epi(x, o), time_ms=dt))
                if i in ex_idx:
                    examples[(li, i)][m] = o
            log(f"  σ={s}: image {i + 1}/{len(test_clean)} done")
        log(f"  σ={s}: done ({len(test_clean)} images x {len(order)} methods)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_image_results.csv"), index=False)
    ref_enl = float(np.mean([m_enl(x, r) for x, r in zip(test_clean, rois)]))

    # ---------- CPU timing of AI models (fair compute comparison with classical filters) ----------
    cpu_rows = []
    if dev.type == "cuda" and ai_run:
        log("\n== CPU timing of AI models (same hardware as classical filters) ==")
        cpu = torch.device("cpu")
        imgs = test_noisy[len(levels) // 2][:args.cpu_timing_images]
        nets = {}
        if "DnCNN" in ai_run:
            nets["DnCNN"] = lambda y, n=copy.deepcopy(dncnn).to(cpu): run_net(n, y, cpu)
        if "Noise2Void" in ai_run:
            nets["Noise2Void"] = lambda y, n=copy.deepcopy(n2v).to(cpu): run_net(n, y, cpu)
        if "DRUNet" in ai_run:
            dcpu = copy.deepcopy(drunet).to(cpu)
            nets["DRUNet"] = lambda y, n=dcpu: drunet_speckle(n, y, dr_sig[len(levels) // 2], cpu)
        for m, fn in nets.items():
            fn(imgs[0])
            t0 = time.perf_counter()
            for y in imgs:
                fn(y)
            cpu_rows.append(dict(method=m, cpu_ms_per_image=(time.perf_counter() - t0) * 1000 / len(imgs),
                                 note="measured"))
        if "DIP" in ai_run:
            k = 20
            t0 = time.perf_counter()
            dip_denoise(imgs[0], k, cpu)
            per_it = (time.perf_counter() - t0) * 1000 / k
            it_mid = dip_iters[len(levels) // 2]
            cpu_rows.append(dict(method="DIP", cpu_ms_per_image=per_it * it_mid,
                                 note=f"extrapolated: {k} iters measured x {it_mid}"))
    cpu_df = pd.DataFrame(cpu_rows)

    # ---------- summaries ----------
    log("\n== Summaries ==")
    g = df.groupby(["level", "method"])
    summ = g[QUALITY_METRICS + ["time_ms"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ["method"] = pd.Categorical(summ["method"], ["Noisy input"] + order, ordered=True)
    summ = summ.sort_values(["level", "method"])
    summ.to_csv(os.path.join(out, "summary_by_level.csv"), index=False)

    pretty = summ[["level", "method"]].copy()
    for met in QUALITY_METRICS:
        pretty[met] = [f"{a:.3f} ± {b:.3f}" if met != "ENL" else f"{a:.1f} ± {b:.1f}"
                       for a, b in zip(summ[f"{met}_mean"], summ[f"{met}_std"])]
    pretty["time (ms)"] = [f"{a:.1f}" for a in summ["time_ms_mean"]]

    overall = df[df.method != "Noisy input"].groupby("method")[QUALITY_METRICS + ["time_ms"]].mean()
    overall = overall.loc[order].reset_index()
    overall.to_csv(os.path.join(out, "summary_overall.csv"), index=False)

    t_sum = pd.DataFrame(dict(method=order,
                              ms_per_image=[df[df.method == m].time_ms.mean() for m in order],
                              device=[methods[m][1] for m in order]))
    info_df = pd.DataFrame([dict(method=k, **v) for k, v in info.items()])
    info_df.to_csv(os.path.join(out, "methods_info.csv"), index=False)

    ranks_all = mean_ranks(df, order)
    best_ai = next((m for m in ranks_all.index if m in ai_run), None)
    best_cl = next((m for m in ranks_all.index if m in cl_run), None)
    fr_df, w_df = (stat_tests(df, order, ai_run, best_cl, levels) if best_cl
                   else (pd.DataFrame(), pd.DataFrame()))
    fr_df.to_csv(os.path.join(out, "stats_friedman.csv"), index=False)
    w_df.to_csv(os.path.join(out, "stats_wilcoxon_vs_best_classical.csv"), index=False)

    # ---------- figures ----------
    plot_metric_bars(df, order, levels, os.path.join(fig_dir, "metrics_bars.png"))
    plot_vs_noise(df, order, levels, os.path.join(fig_dir, "psnr_ssim_vs_noise.png"))
    plot_boxes(df, order, levels, "PSNR", os.path.join(fig_dir, "psnr_boxplots.png"))
    plot_boxes(df, order, levels, "SSIM", os.path.join(fig_dir, "ssim_boxplots.png"))
    plot_time(t_sum, os.path.join(fig_dir, "processing_time.png"), size=S)
    for li in range(len(levels)):
        plot_examples(examples, order, levels, li, os.path.join(fig_dir, f"visual_sigma{levels[li]}.png"))
    plot_training(train_curves, dip_curves, os.path.join(fig_dir, "training_curves.png"))

    # ---------- report (every number below comes from this run) ----------
    rep = ["# BUSI speckle reduction: AI vs classical — results of this run", ""]
    if args.quick:
        rep += ["> **QUICK MODE — smoke test only. Do not report these numbers.**", ""]
    rep += [f"- Device: {dev} ({gpu}); torch {torch.__version__}",
            f"- Images: {len(test_clean)} test / {len(val_clean)} val / {len(train_clean)} train, "
            f"resized to {S}x{S}, grayscale",
            f"- Noise: y = x·(1 + σ·n), n ~ N(0,1), σ ∈ {levels}; identical noisy images for all methods",
            f"- Reference = original BUSI image (already contains native speckle). "
            f"Mean ENL of the reference ROIs: {ref_enl:.1f}", "",
            "## Methods actually run", "", md_table(info_df), "",
            "## Results per noise level (mean ± std over test images)", "", md_table(pretty), "",
            "## Overall means (all noise levels)", "", md_table(overall), ""]
    if len(cpu_df):
        rep += ["## CPU inference time of AI models (same CPU as classical filters)", "",
                md_table(cpu_df, ".1f"), ""]
    rep += ["## Ranking", "",
            f"Mean per-image rank across {', '.join(RANK_METRICS)} (1 = best; ENL excluded because it "
            f"rewards over-smoothing):", "",
            md_table(ranks_all.reset_index().rename(columns={"rank": "mean_rank"})), "",
            f"- **Best AI model:** {best_ai}", f"- **Best classical filter:** {best_cl}", ""]
    if best_ai and best_cl:
        for s in levels:
            a = summ[(summ.level == s) & (summ.method == best_ai)].iloc[0]
            b = summ[(summ.level == s) & (summ.method == best_cl)].iloc[0]
            rep.append(f"- σ={s}: {best_ai} vs {best_cl}: PSNR {a.PSNR_mean:.2f} vs {b.PSNR_mean:.2f} dB, "
                       f"SSIM {a.SSIM_mean:.3f} vs {b.SSIM_mean:.3f}, EPI {a.EPI_mean:.3f} vs {b.EPI_mean:.3f}, "
                       f"ENL {a.ENL_mean:.1f} vs {b.ENL_mean:.1f}, time {a.time_ms_mean:.1f} vs "
                       f"{b.time_ms_mean:.1f} ms")
        rep.append("")
    if len(fr_df):
        rep += ["## Statistics", "", "Friedman test (all methods, paired by image):", "", md_table(fr_df), "",
                f"Wilcoxon signed-rank, each AI model vs best classical ({best_cl}); Holm-corrected over all rows. "
                f"median_diff > 0 means the AI model scores higher:", "", md_table(w_df), ""]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep))
    log("\n".join(rep))
    log(f"\nAll outputs written to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
