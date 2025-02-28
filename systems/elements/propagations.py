import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
from utils import padded_fftnd, unpadded_ifftnd


class ASMPropagation(nn.Module):
    def __init__(self, z, lamb0, ref_idx, Lx, Ly, f_grid, band_limited=True):
        super(ASMPropagation, self).__init__()
        self.z = z
        self.lamb0 = lamb0
        self.ref_idx = ref_idx
        self.Lx = Lx
        self.Ly = Ly
        self.fx, self.fy = f_grid[0], f_grid[1]
        self.band_limited = band_limited
        
    def forward(self, field):
        return asm_propagation(
            input_field=field,
            z=self.z,
            n=self.ref_idx,
            lamb0=self.lamb0,
            Lx=self.Lx,
            Ly=self.Ly,
            fx=self.fx,
            fy=self.fy,
            band_limited=self.band_limited
        )

def asm_propagation(input_field, z, n: float, lamb0: Tensor, Lx: float, Ly: float, fx: Tensor = None, fy: Tensor = None, band_limited: bool = True):
    propagator = asm_propagator(input_field, z, n, lamb0, Lx, Ly, fx, fy, band_limited)
    return field_propagate(input_field, propagator)

def asm_propagator(input_field, z, n: float, lamb0: Tensor, Lx: float, Ly: float, fx: Tensor = None, fy: Tensor = None, band_limited: bool = True):
    H, W = input_field.shape[-2:]
    transform_dims = -np.arange(1, 2+1)[::-1].tolist()
    # input phase : B, C,.. , H, W, ; complex tensor
    # lamb: Tensor, C,
    # Lx & Ly ; Physical size of image space.
    
    k = 2*np.pi*n / lamb0
    lamb = lamb0/n
    
    dfx = 1 / Lx
    dfy = 1 / Ly
    
    # d : inverse of sampling rate. sampling rate : 1/Lx
    # define those with frequency instead of wavevector
    if fx == None or fy == None:
        fx = np.fft.fftshift(np.fft.fftfreq(H, 1/dfx))
        fy = np.fft.fftshift(np.fft.fftfreq(W, 1/dfy))
        fx, fy = np.meshgrid(fx, fy)
        fx, fy = torch.from_numpy(fx), torch.from_numpy(fy)
    
    fz = (1/lamb) ** 2 - (fx**2 + fy**2)
    fz = torch.sqrt(torch.maximum(fz, 0))
    kz = 2 * np.pi * fz
    propagator = torch.exp(1j * kz * z)
    
    ### BL-ASM ; Antialiasing
    if band_limited:
        f_limit_px = ((1/(2*dfx)) ** (-2) * z**2 + 1) ** (-1/2) / lamb
        f_limit_nx = ( -(1/(2*dfx)) ** (-2) * z**2 + 1) ** (-1/2) / lamb
        f0_x = (1/2) * ((Lx * f_limit_px) - (Lx * f_limit_nx))
        f_width_x = (1/2) * ((Lx * f_limit_px) + (Lx * f_limit_nx))        
        
        f_limit_py = ((1/(2*dfy)) ** (-2) * z**2 + 1) ** (-1/2) / lamb
        f_limit_ny = ( -(1/(2*dfy)) ** (-2) * z**2 + 1) ** (-1/2) / lamb
        f0_y = (1/2) * ((Lx * f_limit_py) - (Ly * f_limit_ny))
        f_width_y = (1/2) * ((Lx * f_limit_py) + (Ly * f_limit_ny))  
        
        fx_max = f_width_x / 2
        fy_max = f_width_y / 2
        
        H_filter_x = torch.abs(torch.abs(fx-f0_x)) <= fx_max
        H_filter_y = torch.abs(torch.abs(fy-f0_y)) <= fy_max
        H_filter = H_filter_x * H_filter_y
    
        propagator = propagator * H_filter    
    propagator = propagator[None, None, :, :] # H, W -> 1, 1, H, W
    return torch.fft.ifftshift(propagator,dim=transform_dims)
    
def field_propagate(input_field, propagator, ndim):
    # input_field : B, C, ..., H, W
    # propagator : 1, 1, H, W
    for n in range(ndim-2):
        propagator = propagator.unsqueeze(0)
    field_freq = padded_fftnd(input_field, n=ndim) # in this function, we do not use rfft to maintain the spatial size
    field_filtered = torch.fft.ifftshift(torch.fft.fftshift(field_freq) * propagator)
    out = unpadded_ifftnd(field_filtered, n=ndim)
    return out
    