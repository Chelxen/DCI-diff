"""3D cone-beam operator for the GE cardiac CT projections (LEAP, GPU-resident torch tensors).

Geometry (GE scanner, curved detector):
  * projections: 984 views x 256 detector rows x 821 detector columns, angles linspace(0, 360, 984)
  * source-to-object 625.61 mm, source-to-detector 1097 mm
  * reconstruction grid: 512 x 512 x 512, voxel 0.85 mm (in-plane) x 0.28 mm (z), centred on the isocentre

With this geometry the full-scan FDK reproduces the reference volumes (clamped at 0) to ~1e-7, and the
limited-angle FBP inputs are the FBP of the first n views (n = 246 for 90 deg, 328 for 120 deg).

Volumes are [Z, Y, X] float32 torch tensors, projections are [views, rows, cols].
"""
import math

import numpy as np
import torch
from leapctype import tomographicModels

N_VIEWS, N_ROWS, N_COLS = 984, 256, 821
ROW_PITCH = 1.09                      # mm, detector row spacing
COL_PITCH = 0.8224 * 1097 / 821       # mm, detector column spacing (arc length on the curved detector)
SOD, SDD = 625.61, 1097.0             # mm
VOXEL_XY, VOXEL_Z, NXY, NZ = 0.85, 0.28, 512, 512
LIMITED_VIEWS = {90: 246, 120: 328}   # number of measured views per angular range


def full_angles():
    """Physical projection angles (degrees) of all 984 views."""
    return np.linspace(0, 360, N_VIEWS, dtype=np.float32)


def load_projections(path):
    """Memory-mapped projection array of shape (984, 256, 821)."""
    proj = np.load(path, mmap_mode='r')
    if proj.shape != (N_VIEWS, N_ROWS, N_COLS):
        raise ValueError(f'expected projections of shape {(N_VIEWS, N_ROWS, N_COLS)}, got {proj.shape}')
    return proj


def make_leap(angles, nz, gpu, rows=slice(0, N_ROWS)):
    """LEAP model for the given view angles, an nz-slice volume and a contiguous range of detector rows."""
    ct = tomographicModels()
    ct.set_gpu(gpu)
    ct.set_conebeam(len(angles), rows.stop - rows.start, N_COLS, ROW_PITCH, COL_PITCH,
                    0.5 * (N_ROWS - 1) - rows.start, 0.5 * (N_COLS - 1),
                    np.ascontiguousarray(angles, dtype=np.float32), SOD, SDD)
    ct.set_curvedDetector()
    ct.set_volume(NXY, NXY, nz, voxelWidth=VOXEL_XY, voxelHeight=VOXEL_Z)
    return ct


class ConeBeamOperator:
    """Cone-beam forward operator A on one CUDA device.

    The model volume has `nz` slices: `pad` DC-only slices on each side of the slices handled by the
    2D prior. Full-volume reconstruction uses nz = 512 + 2 * pad, because the outer detector rows see
    beyond the 512-slice stack. `rows` restricts the detector rows (used for the slab experiments).
    """

    def __init__(self, view_indices, pad, device, nz=None, rows=None):
        self.device = torch.device(device)
        self.view_indices = np.asarray(view_indices)
        self.pad = int(pad)
        self.nz = int(nz) if nz is not None else NZ + 2 * self.pad
        self.rows = rows if rows is not None else slice(0, N_ROWS)
        self.ct = make_leap(full_angles()[self.view_indices], self.nz, self.device.index, self.rows)
        self.vol_shape = (self.nz, NXY, NXY)
        self.proj_shape = (len(self.view_indices), self.rows.stop - self.rows.start, N_COLS)

    def fmult(self, x):
        g = torch.zeros(self.proj_shape, dtype=torch.float32, device=self.device)
        self.ct.project(g, x.reshape(self.vol_shape).contiguous())
        return g

    def ftran(self, g):
        f = torch.zeros(self.vol_shape, dtype=torch.float32, device=self.device)
        self.ct.backproject(g.contiguous(), f)
        return f

    def measurements(self, proj):
        b = np.ascontiguousarray(proj[self.view_indices][:, self.rows], dtype=np.float32)
        return torch.from_numpy(b).to(self.device)

    def split(self, vol):
        """Model volume -> (low pad, prior slices, high pad)."""
        p = self.pad
        return vol[:p], vol[p:self.nz - p], vol[self.nz - p:]

    @staticmethod
    def join(low, center, high):
        return torch.cat([low, center, high], 0)


def limited_fbp_input(proj, angle, nz, device):
    """Limited-angle FBP condition image (FBP of the first n views, angles labelled linspace(0, angle, n),
    clamped at 0) on an nz-slice volume centred on the isocentre."""
    n = LIMITED_VIEWS[angle]
    ct = make_leap(np.linspace(0, angle, n, dtype=np.float32), nz, torch.device(device).index)
    g = torch.from_numpy(np.ascontiguousarray(proj[:n], dtype=np.float32)).to(device)
    f = torch.zeros((nz, NXY, NXY), dtype=torch.float32, device=device)
    ct.FBP(g, f)
    return f.clamp_(min=0)


def slab_rows(nslab):
    """Detector rows whose rays stay inside a central slab of `nslab` slices for every point of the
    512 x 512 field of view (worst-case source distance SOD + sqrt(2) * FOV / 2, half-pixel footprint).
    Restricting A to these rows makes the slab sub-problem exact."""
    half_z = 0.5 * nslab * VOXEL_Z
    d_max = SOD + math.sqrt(2) * NXY * VOXEL_XY / 2
    h = int(math.floor(half_z * SDD / d_max / ROW_PITCH - 0.5))
    c = 0.5 * (N_ROWS - 1)
    return slice(int(math.ceil(c - h - 0.5 + 1e-6)), int(math.floor(c + h + 0.5 - 1e-6)) + 1)
