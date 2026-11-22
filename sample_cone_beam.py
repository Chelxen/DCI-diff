#!/usr/bin/env python3
"""Limited-angle reconstruction of a whole GE volume: 2D diffusion prior + measured 3D cone-beam DC.

The condition image (limited-angle FBP) and the DC measurements are both computed from the measured
cone-beam projections; the reference volume (--gt) is only used for evaluation.

Example:
  python sample_cone_beam.py --projections data/projections/patient<ID>_adjusted.npy --angle 90 \
      --checkpoint checkpoints/GE_90deg.pth --gt data/reference/<ID>/full_npy --device cuda:0
"""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from dataconsistency.cone_beam import ConeBeamOperator, LIMITED_VIEWS, limited_fbp_input, load_projections
from dataconsistency.cone_beam_sampler import (load_net, load_volume, relative_residual, sample_volume,
                                               volume_metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--projections', required=True, help='projection .npy of shape (984, 256, 821)')
    p.add_argument('--angle', type=int, choices=sorted(LIMITED_VIEWS), default=90,
                   help='angular range of the measured views (first 246 / 328 views)')
    p.add_argument('--checkpoint', required=True, help='network weights (.pth)')
    p.add_argument('--config', default='config/GE_test.json', help='network / noise-schedule config')
    p.add_argument('--gt', default=None, help='reference volume (.npy or directory of slice .npy) for metrics')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=25, help='sampling steps')
    p.add_argument('--dc-steps', type=int, default=25, help='DC is applied at the last DC-steps steps')
    p.add_argument('--cg-iter', type=int, default=2, help='CG iterations per DC step (0: no DC)')
    p.add_argument('--lam', type=float, default=0.0, help='weight of the proximity term to the prior estimate')
    p.add_argument('--no-nonneg', dest='nonneg', action='store_false',
                   help='do not clamp the DC output at 0')
    p.add_argument('--pad', type=int, default=130, help='DC-only slices added on each side in z')
    p.add_argument('--chunk', type=int, default=16, help='slices per network forward pass')
    p.add_argument('--seed', type=int, default=1688)
    p.add_argument('--output-dir', default='results/cone_beam')
    p.add_argument('--name', default=None, help='output file prefix')
    a = p.parse_args()

    device = torch.device(a.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or f'{Path(a.projections).stem}_{a.angle}deg_s{a.steps}_dc{a.dc_steps}_cg{a.cg_iter}'
    log = lambda m: print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

    wall0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    proj = load_projections(a.projections)
    A = ConeBeamOperator(np.arange(LIMITED_VIEWS[a.angle]), a.pad, device)
    b = A.measurements(proj)
    low, cond, high = A.split(limited_fbp_input(proj, a.angle, A.nz, device))
    cond, low, high = cond[:, None].contiguous(), low.clone(), high.clone()
    net = load_net(a.config, a.checkpoint, device)

    t0 = time.time()
    vol, (low, high), stats = sample_volume(net, A, cond, b, (low, high), steps=a.steps, dc_steps=a.dc_steps,
                                            cg_iter=a.cg_iter, lam=a.lam, nonneg=a.nonneg, seed=a.seed,
                                            chunk=a.chunk, log=log)
    stats['sampling_seconds'] = time.time() - t0
    del net

    rec = dict(name=name, projections=str(a.projections), angle=a.angle, checkpoint=a.checkpoint,
               settings=dict(steps=a.steps, dc_steps=a.dc_steps, cg_iter=a.cg_iter, lam=a.lam, nonneg=a.nonneg,
                             pad=a.pad, chunk=a.chunk, seed=a.seed),
               **stats, relative_residual=relative_residual(A, A.join(low, vol, high), b))
    if a.gt is not None:
        gt = torch.from_numpy(load_volume(a.gt)).to(device)
        rec['input_metrics'] = volume_metrics(cond[:, 0], gt)
        rec['metrics'] = volume_metrics(vol, gt)
    rec['gpu'] = torch.cuda.get_device_name(device)
    rec['peak_gpu_memory_MiB'] = round(torch.cuda.max_memory_allocated(device) / 2 ** 20)
    rec['peak_host_memory_MiB'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    rec['wall_seconds'] = time.time() - wall0
    np.save(out / f'{name}_volume.npy', vol.cpu().numpy().astype(np.float32))
    (out / f'{name}.json').write_text(json.dumps(rec, indent=2))
    log(json.dumps(rec))


if __name__ == '__main__':
    main()
