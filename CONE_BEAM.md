# Cone-beam data consistency

This part reconstructs whole limited-angle GE cardiac CT volumes from the **measured** cone-beam
projections. The trained 2D conditional diffusion model is used as a slice-wise prior, and every
sampling step enforces consistency with the measured projections through a 3D cone-beam operator.
The reference volume is only used to compute metrics.

## Requirements

On top of `requirements.txt`, install [LEAP](https://github.com/LLNL/LEAP) (`leapctype`, tested with
1.26) with CUDA support. The scripts use one CUDA GPU; a full 512-slice volume needs about 26 GB of
GPU memory with the default batch of 16 slices.

## Data

| Input | Shape | Use |
|---|---|---|
| Projections `patient<ID>_adjusted.npy` | 984 views x 256 rows x 821 columns | condition image and DC measurements |
| Reference volume (`.npy` or a directory of slice `.npy` files) | 512 x 512 x 512 | metrics only |

The scanner geometry is defined in `dataconsistency/cone_beam.py`: curved detector, source-to-object
625.61 mm, source-to-detector 1097 mm, 984 views over 360 degrees, and a 512^3 grid with
0.85 mm x 0.85 mm x 0.28 mm voxels. A 90 degree scan uses the first 246 views and a 120 degree scan
the first 328 views. The condition image is the FBP of those views, computed from the projections.

## Method

At sampling step t the network predicts the clean volume slice by slice. The predicted volume z is
then corrected by a few conjugate-gradient (CG) iterations on

    min_x  0.5 ||A x - b||^2 + 0.5 * lam * ||x - z||^2

started at z, followed by clamping at 0. Here A is the cone-beam projector of the measured views and
b the measured projections. The DDPM posterior then gives the next sample. Sampling uses 25 of the
500 training timesteps.

Because the cone beam reaches beyond the 512 reconstructed slices, the DC volume has 130 extra
slices on each side in z. These slices are only updated by DC and are discarded at the end.

Default setting: 25 sampling steps, DC at all 25 steps, 2 CG iterations per DC step, lam = 0.
More CG iterations fit the measurement noise and reduce image quality.

## Usage

Full-volume reconstruction:

```bash
python sample_cone_beam.py \
    --projections data/projections/patient<ID>_adjusted.npy \
    --gt data/reference/<ID>/full_npy \
    --angle 90 --checkpoint <weights.pth> --device cuda:0
```

It writes `<name>_volume.npy` (512 x 512 x 512) and `<name>.json` with the settings, metrics,
projection residual, timing and peak memory. `--gt` is optional.

## Metrics

Both volumes are clipped to [0.3, 1.3] and mapped to [0, 1]. PSNR uses a data range of 1, and SSIM
follows the scikit-image defaults (7 x 7 uniform window). Values are averaged over slices.

## Main options

| Option | Default | Meaning |
|---|---|---|
| `--steps` | 25 | sampling steps |
| `--dc-steps` | 25 | DC is applied at the last `dc-steps` steps |
| `--cg-iter` | 2 | CG iterations per DC step (0 disables DC) |
| `--lam` | 0 | weight of the proximity term to the network estimate |
| `--no-nonneg` | off | disable clamping the DC output at 0 |
| `--pad` | 130 | DC-only slices on each side in z |
| `--chunk` | 16 | slices per network forward pass (lower it to save memory) |
| `--seed` | 1688 | random seed |

## Code

| File | Content |
|---|---|
| `dataconsistency/cone_beam.py` | geometry, cone-beam operator, limited-angle FBP |
| `dataconsistency/cone_beam_sampler.py` | network loading, sampler with DC, CG solver, metrics |
| `sample_cone_beam.py` | full-volume reconstruction |
