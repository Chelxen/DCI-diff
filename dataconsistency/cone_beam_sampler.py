"""Diffusion sampling with a slice-wise 2D prior and 3D cone-beam data consistency (DC).

At every DC step the predicted clean volume z (prior slices plus the DC-only padding slices) is
replaced by CG iterations on   min_x 0.5 ||A x - b||^2 + 0.5 lam ||x - z||^2,   started at z,
followed by a non-negativity clamp. The DDPM posterior then maps it to the next step.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.networks import define_G


def time_indices(total, steps):
    """`steps` evenly spaced timesteps of the training schedule, in sampling order."""
    if not 2 <= steps <= total:
        raise ValueError('sampling steps must be between 2 and the number of training timesteps')
    return np.linspace(0, total - 1, steps).round().astype(int).tolist()[::-1]


def cg_solve(A, z, b, lam, max_iter, rtol=1e-5):
    """CG on (A^T A + lam I) x = A^T b + lam z, started at x = z (lam = 0: CG on the normal equations)."""
    x = z.clone()
    r = A.ftran(b - A.fmult(x))
    p = r.clone()
    rr = (r * r).sum()
    threshold = rr * rtol ** 2
    n = 0
    for _ in range(max_iter):
        if rr <= threshold or rr <= torch.finfo(rr.dtype).tiny:
            break
        Ap = A.ftran(A.fmult(p)) + lam * p
        denom = (p * Ap).sum()
        if not torch.isfinite(denom) or denom <= 0:
            raise FloatingPointError('CG lost positive curvature')
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = (r * r).sum()
        p = r + (rr_new / rr) * p
        rr = rr_new
        n += 1
    if not torch.isfinite(x).all():
        raise FloatingPointError('non-finite CG output')
    return x, n


def load_net(config, checkpoint, device):
    """Denoising network and noise schedule from a training config and checkpoint."""
    opt = json.loads(Path(config).read_text())
    opt.update(phase='test', uncertainty_train=False, distributed=False, gpu_ids=[])
    net = define_G(opt).to(device).eval()
    net.set_new_noise_schedule(opt['model']['beta_schedule']['train'], device)
    state = torch.load(checkpoint, map_location='cpu')
    net.load_state_dict({(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}, strict=True)
    return net


@torch.no_grad()
def predict_clean(net, cond, x, a, chunk):
    """x0 estimate for all slices, running the 2D network in chunks of `chunk` slices."""
    out = torch.empty_like(x)
    level = a.sqrt().reshape(1, 1)
    for s in range(0, x.shape[0], chunk):
        e = slice(s, s + chunk)
        eps = net.denoise_fn(torch.cat([cond[e], x[e]], 1), level.expand(x[e].shape[0], 1))[0]
        out[e] = (x[e] - (1 - a).sqrt() * eps) / a.sqrt()
    return out


@torch.no_grad()
def sample_volume(net, A, cond, b, pads, steps=25, dc_steps=25, cg_iter=2, lam=0.0, nonneg=True,
                  seed=1688, chunk=16, log=None):
    """Respaced ancestral DDPM sampling with cone-beam DC at the last `dc_steps` steps.

    cond: [S, 1, H, W] limited-angle condition of the S prior slices.
    pads: (low, high) DC-only slices of the model volume (initialised from the FBP input).
    Returns the S prior slices [S, H, W], the final pads and timing statistics.
    """
    rng = torch.Generator(device=cond.device).manual_seed(seed)
    x = torch.randn(cond.shape, device=cond.device, generator=rng)
    indices = time_indices(net.num_timesteps, steps)
    low, high = pads
    cg_total = 0
    t_net = t_dc = 0.0
    for k, t in enumerate(indices):
        prev = indices[k + 1] if k + 1 < len(indices) else -1
        a = net.alphas_cumprod[t]
        ap = net.alphas_cumprod[prev] if prev >= 0 else torch.ones_like(a)
        t0 = time.time()
        clean = predict_clean(net, cond, x, a, chunk)
        torch.cuda.synchronize()
        t_net += time.time() - t0
        if cg_iter > 0 and len(indices) - k <= dc_steps:
            t0 = time.time()
            sol, n = cg_solve(A, A.join(low, clean[:, 0], high), b, lam, cg_iter)
            if nonneg:
                sol = sol.clamp_(min=0)
            low, center, high = A.split(sol)
            clean = center[:, None].contiguous()
            del sol, center
            torch.cuda.synchronize()
            t_dc += time.time() - t0
            cg_total += n
        beta = 1 - a / ap
        x = ap.sqrt() * beta / (1 - a) * clean + (a / ap).sqrt() * (1 - ap) / (1 - a) * x
        if prev >= 0:
            x = x + (beta * (1 - ap) / (1 - a)).clamp_min(0).sqrt() * torch.randn(
                x.shape, device=x.device, generator=rng)
        del clean
        if log is not None:
            log(f'step {k + 1}/{len(indices)}  net {t_net:.1f}s  dc {t_dc:.1f}s  cg {cg_total}')
    stats = dict(cg_total=cg_total, net_seconds=t_net, dc_seconds=t_dc)
    return x[:, 0], (low, high), stats


# ----------------------------------------------------------------------------------------------- metrics
PAPER_WINDOW = (0.3, 1.3)


def _ssim(x, y, data_range=1.0, win=7):
    """skimage-equivalent SSIM (7x7 uniform window, sample covariance, K1 = 0.01, K2 = 0.03) per slice,
    for float64 tensors [B, H, W]."""
    x, y = x[:, None], y[:, None]
    f = lambda t: F.avg_pool2d(t, win, stride=1)
    ux, uy = f(x), f(y)
    uxx, uyy, uxy = f(x * x), f(y * y), f(x * y)
    cov = win * win / (win * win - 1.0)
    vx, vy, vxy = cov * (uxx - ux * ux), cov * (uyy - uy * uy), cov * (uxy - ux * uy)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return s.mean(dim=(1, 2, 3))


@torch.no_grad()
def volume_metrics(pred, gt, window=PAPER_WINDOW, batch=32):
    """Slice-averaged PSNR / SSIM / RMSE after clipping both volumes to `window` and mapping it to [0, 1]
    (data range 1). pred, gt: [S, H, W] tensors on the same device."""
    lo, hi = window
    norm = lambda v: ((v.float().clamp(lo, hi) - lo) / (hi - lo)).double()
    psnr, ssim, rmse = [], [], []
    for s in range(0, pred.shape[0], batch):
        x, g = norm(pred[s:s + batch]), norm(gt[s:s + batch])
        mse = ((x - g) ** 2).mean((1, 2))
        psnr += (10 * torch.log10(1 / mse)).tolist()
        ssim += _ssim(g, x).tolist()
        rmse += mse.sqrt().tolist()
    return dict(psnr=float(np.mean(psnr)), ssim=float(np.mean(ssim)), rmse=float(np.mean(rmse)))


@torch.no_grad()
def relative_residual(A, vol, b):
    """||A vol - b|| / ||b|| on the measured views."""
    return float((A.fmult(vol) - b).norm() / b.norm())


def load_volume(path):
    """A 3D .npy volume, or a directory of per-slice .npy files (sorted by name), as a float32 array."""
    p = Path(path)
    if p.is_dir():
        return np.stack([np.load(f) for f in sorted(p.glob('*.npy'))]).astype(np.float32)
    return np.load(p).astype(np.float32)
