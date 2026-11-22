################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# DOLCE project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Adapted from DOLCE (https://github.com/wustl-cig/DOLCE) for DCI-diff:
# fan-beam data consistency module for CT reconstruction.
################################################################################

import os
import sys
import numpy as np
import torch
from leaptorch import Projector


def create_circular_mask(h, w, center=None, radius=None):
    """
    Generate a circular mask to black out none ROI
    return: mask, numpy array, (H,W)
    """
    if center is None:
        center = (int(h/2)-0.5, int(w/2)-0.5)
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


class CTClass:
    """
    Utilities for CTClass using LEAP ToolBox.
    
    :param projector: projector geometry (forward_project=True)
    :param size: image size, default (256, 256)
    :param device: if specified, the device to create the geometry on.
    :param gpu_index: if specified, the gpu_index to create the geometry on.
    :param f_fov: if specified, the region of interest to sample within a image. 
    """

    def __init__(self, projector, 
                 size=(256, 256), 
                 device=None, 
                 gpu_index=None, 
                 f_fov=None):
        self.M, self.N = size
        self.projector = projector
        
        self.backprojector = Projector(
            forward_project=False, 
            use_static=projector.use_static, 
            use_gpu=projector.use_gpu, 
            gpu_device=projector.gpu_device, 
            batch_size=projector.batch_size
        )
        self.backprojector.leapct = projector.leapct
        self.backprojector.param_id_t = projector.param_id_t
        self.backprojector.vol_data = projector.vol_data
        self.backprojector.proj_data = projector.proj_data
        
        if f_fov is not None:
            self.img_fov = f_fov.to(device)
        else:
            self.img_fov = None
        self.device, self.gpu_index = device, gpu_index

    def train(self):
        """Set projector to training mode"""
        self.projector.train()

    def eval(self):
        """Set projector to evaluation mode"""
        self.projector.eval()

    def xinit(self, sino):
        """
        Initialize image from sinogram using backprojection
        :param sino: sinogram data, (N, Views, Rows, Cols)
        :return: initial image, (N, C, H, W)
        """
        x = self.ftran(sino)
        return x
    
    def grad(self, x, sino):
        """
        Performs gradient of ||Ax-b||_2^2
        :param x: image, (N, C, H, W)
        :param sino: sinogram data, (N, Views, Rows, Cols)
        :return: gradient wrt image x, (N, C, H, W)
        """
        grad = self.ftran(self.fmult(x) - sino)
        return grad
        
    def cgrad_ori(self, ipt, sino_sample, device, lam=1e5, max_iter=100, tol=1e-10):
        """
        Performs CG algorithm to solve: argmin_x ||Ax-b||^2 + lambda||x-u||^2
        
        :param ipt: dict with key "sample_data", initial/prior image (N, C, H, W)
        :param sino_sample: sinogram data, (N, Views, Rows, Cols)
        :param device: torch device
        :param lam: regularization parameter, default 1e5
        :param max_iter: maximum iterations, default 100
        :param tol: convergence tolerance, default 1e-10
        :return: reconstructed image x, (N, C, H, W)
        """
        z_k = ipt["sample_data"]
        Aty = self.ftran(sino_sample)
        x = torch.zeros_like(z_k)
        rhs = Aty + lam * z_k - self.AtA(x, lam)
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(r.conj()*r)
        
        while i < max_iter and rTr > tol: 
            Ap = self.AtA(p, lam)
            alpha = rTr / torch.sum(p.conj()*Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.sum(r.conj()*r)
            beta = rTrNew / rTr
            p = r + beta * p
            i += 1
            rTr = rTrNew
        return x

    def cgrad(self, ipt, sino_sample, device, lam=1e5, max_iter=100, tol=1e-10):
        """
        Performs CG algorithm to solve: argmin_x ||Ax-b||^2 (pure data consistency)

        :param ipt: dict with key "sample_data", initial image (N, C, H, W)
        :param sino_sample: sinogram data, (N, Views, Rows, Cols)
        :param device: torch device
        :param lam: regularization parameter (not used in pure DC, but kept for compatibility)
        :param max_iter: maximum iterations, default 100
        :param tol: convergence tolerance, default 1e-10
        :return: reconstructed image x, (N, C, H, W)
        """
        z_k = ipt["sample_data"]
        Aty = self.ftran(sino_sample)
        x = z_k.clone()  # Use diffusion sample as initial value
        rhs = Aty  # Pure data consistency: (A^T A)x = A^T b
        r = rhs - self.AtA(x, 0)  # r = rhs - (A^T A)x, with lam=0
        p = r
        i = 0
        rTr = torch.sum(r.conj()*r)

        while i < max_iter and rTr > tol:
            Ap = self.AtA(p, 0)  # AtA without regularization
            alpha = rTr / torch.sum(p.conj()*Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = torch.sum(r.conj()*r)
            beta = rTrNew / rTr
            p = r + beta * p
            i += 1
            rTr = rTrNew
        
        return x
    
    def apgm(self, ipt, sino_sample, rho, device, max_iter=30, lr=5e-6):
        """
        Performs APGM (Accelerated Proximal Gradient Method) algorithm
        
        :param ipt: dict with key "sample_data", initial image (N, C, H, W)
        :param sino_sample: sinogram data, (N, Views, Rows, Cols)
        :param rho: regularization parameter
        :param device: torch device
        :param max_iter: maximum iterations, default 30
        :param lr: learning rate, default 5e-6
        :return: reconstructed image x, (N, C, H, W)
        """
        accelerate = True
        x = ipt["sample_data"].detach().cpu().clone().to(device)
        z = ipt["sample_data"].detach().cpu().clone().to(device)
        s = x
        t = torch.tensor([1.]).float().to(device)
        
        for _ in range(max_iter):
            grad = self.grad(s, sino_sample)
            xnext = s - lr * grad - rho * (s - z)
            # acceleration
            if accelerate:
                tnext = 0.5 * (1 + torch.sqrt(1 + 4*t*t))
            else:
                tnext = 1
            s = xnext + ((t-1)/tnext) * (xnext - x)
            # update
            t = tnext
            x = xnext
        return x    

    def prox_solver(self, ipt, sino_sample, rho, solver_mode, device, **kwargs):
        """
        Proximal solver for: argmin_x ||Ax-b||_2^2 + ||x-u||^2_2
        
        :param ipt: dict with key "sample_data", initial image (N, C, H, W)
        :param sino_sample: sinogram data, (N, Views, Rows, Cols)
        :param rho: regularization parameter
        :param solver_mode: 'apgm' or 'cgrad'
        :param device: torch device
        :param kwargs: additional parameters for solver
        :return: reconstructed image x, (N, C, H, W)
        """
        if solver_mode == 'apgm':
            max_iter = kwargs.get('max_iter', 30)
            lr = kwargs.get('lr', 5e-6)
            x = self.apgm(ipt, sino_sample, rho, device, max_iter=max_iter, lr=lr)
        elif solver_mode == 'cgrad':
            lam = kwargs.get('lam', 2.5e5)
            max_iter = kwargs.get('max_iter', 100)
            tol = kwargs.get('tol', 1e-10)
            x = self.cgrad(ipt, sino_sample, device, lam=lam, max_iter=max_iter, tol=tol)
        else:
            raise ValueError(f"Unknown solver_mode: {solver_mode}")
        return x
    
    def data_fidelity(self, x, sino):
        """
        Estimate of ||Ax-b||_2^2 (data fidelity term)
        
        :param x: image, (N, C, H, W)
        :param sino: sinogram data, (N, Views, Rows, Cols)
        :return: data fitting error (scalar)
        """
        data_fit = 0.5 * torch.pow(torch.abs(self.fmult(x) - sino), 2).mean()
        return data_fit
    

    def AtA(self, x, lam=0.01):
        """
        Implementation of AtA in Conjugate gradient method 
        :param x: image, (N, C, H, W)
        :param lam: regularization parameter
        :return: (A^T A + lambda*I) * x, (N, C, H, W)
        """        
        return self.ftran(self.fmult(x)) + lam * x

    def fmult(self, x):
        """
        Implementation of Ax (forward projection)
        :param x: image, (N, C, H, W)
        :return: sinogram b, (N, Views, Rows, Cols)
        """           
        sino = self.projector(x)
        # CRITICAL: Return a clone to avoid modifying internal buffers
        # LEAP Projector may return internal buffers that get overwritten
        return sino.clone()    

    def ftran(self, sino, clip=False):
        """
        Implementation of A^T b (backward projection)
        :param sino: sinogram data, (N, Views, Rows, Cols)
        :param clip: whether to clip negative values, default False
        :return: image x, (N, C, H, W)
        """         
        x = self.backprojector(sino)
        # CRITICAL: Return a clone to avoid modifying internal buffers
        # LEAP Projector may return internal buffers that get overwritten
        x = x.clone()
        if clip:
            x[x <= 0] = 0
        return x

    def apply_data_consistency(self, x_sample, sino_obs, method='cgrad', 
                               lam=1e5, rho=None, max_iter=None, **kwargs):
        """
        Apply data consistency to a sample from diffusion model
        
        :param x_sample: sample from diffusion model, (N, C, H, W)
        :param sino_obs: observed sinogram data, (N, Views, Rows, Cols)
        :param method: 'cgrad', 'cgrad_pure_dc' or 'apgm', default 'cgrad'
        :param lam: regularization parameter for cgrad, default 1e5
        :param rho: regularization parameter for apgm
        :param max_iter: maximum iterations
        :param kwargs: additional parameters
        :return: data-consistent image, (N, C, H, W)
        """
        ipt = {"sample_data": x_sample}
        
        if method == 'cgrad':
            if max_iter is None:
                max_iter = kwargs.get('max_iter', 100)
            tol = kwargs.get('tol', 1e-10)
            return self.cgrad(ipt, sino_obs, self.device, lam=lam,
                            max_iter=max_iter, tol=tol)
        elif method == 'cgrad_pure_dc':
            if max_iter is None:
                max_iter = kwargs.get('max_iter', 100)
            tol = kwargs.get('tol', 1e-10)
            return self.cgrad_pure_dc(ipt, sino_obs, self.device,
                                    max_iter=max_iter, tol=tol)
        elif method == 'apgm':
            if rho is None:
                rho = kwargs.get('rho', 0.1)
            if max_iter is None:
                max_iter = kwargs.get('max_iter', 30)
            lr = kwargs.get('lr', 5e-6)
            return self.apgm(ipt, sino_obs, rho, self.device, 
                           max_iter=max_iter, lr=lr)
        else:
            raise ValueError(f"Unknown method: {method}")


def set_CTClass(device, gpu_index=None, use_cuda=True, 
                batch_size=1, use_static=False, use_fov=True, 
                param_fn=None, rank=0, logger=None):
    """
    Initialize CT geometry using LEAP.
    Similar to the original CTClass.py but with fixed Projector API usage.
    
    :param device: torch device
    :param gpu_index: GPU index (deprecated, use device instead)
    :param use_cuda: whether to use CUDA
    :param batch_size: batch size for projection
    :param use_static: whether to use static memory
    :param use_fov: whether to use field of view masking
    :param param_fn: path to geometry parameter file
    :param rank: process rank for distributed training
    :param logger: logger object
    :return: CTClass instance
    """    
    ################################################################################
    # 1. Read or simulate F and G using LEAP
    ################################################################################
    # read arguments
    if param_fn is None:
        # Try to find default parameter file
        default_paths = [
            "./geo_config/XCAT_fan_90.txt"
        ]
        param_fn = None
        for path in default_paths:
            if os.path.isfile(path):
                param_fn = path
                break
        
        if param_fn is None:
            raise FileNotFoundError(
                f"Parameter file not found. Please specify param_fn. "
                f"Tried: {default_paths}"
            )

    # initialize projector and load parameters
    proj = Projector(
        forward_project=True,
        use_static=use_static, 
        use_gpu=use_cuda, 
        gpu_device=device, 
        batch_size=batch_size
    )
    
    if logger is not None:
        logger.info(f"Param setting: {param_fn}")
    
    proj.load_param(param_fn)
  
    if rank == 0:
        proj.print_param()
    
    dimz, dimy, dimx = proj.get_volume_dim()
    views, rows, cols = proj.get_projection_dim()
    M, N = dimz, dimx
    
    # set mask for field of view
    if use_fov:
        f_fov = create_circular_mask(N, N).reshape(1, M, N, N).astype(np.float32)
        f_fov = torch.from_numpy(f_fov)
        if rank == 0 and logger is not None:
            logger.info("Field of view masking is used")
    else:
        f_fov = None
    
    return CTClass(proj, (M, N), device, gpu_index, f_fov)

