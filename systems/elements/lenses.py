import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
from utils import optical_fft

"""
    1. Multiplicative ..
    2. Phase mask
    3. fflens
"""
class MultConvergingLens(nn.Module):
    def __init__(self, grid, lamb0, ref_idx, focal_length, NA, epsilon=np.finfo(np.float32).eps):
        super(MultConvergingLens, self).__init__()
        self.grid = grid
        self.lamb0 = lamb0
        self.ref_idx = ref_idx
        self.focal_length = focal_length
        self.NA = NA
        self.epsilon = epsilon
        
        
    
    def forward(self, field: Tensor):
        """
        Args:
            field (Tensor; B, C, H, W): INPUT SCALAR FIELD
            C : chromatic dimension.
        """
        return multiplicative_phase_lens(
            field=field,
            grid=self.grid,
            lamb0=self.lamb0,
            ref_idx=self.ref_idx,
            f=self.focal_length,
            NA=self.NA,
            epsilon=self.epsilon
        )
        
        

class FFLens(nn.Module):
    def __init__(self, grid, f_grid, lamb0, ref_idx, focal_length, NA):
        super(FFLens, self).__init__()
        self.grid = grid
        self.f_grid = f_grid
        self.lamb0 = lamb0
        self.ref_idx = ref_idx
        self.focal_length = focal_length
        self.NA = NA
        
    def forward(self, field):
        """
        Args:
            field (Tensor; B, C, H, W): INPUT SCALAR FIELD
        """
        return fflens(
            field=field,
            grid=self.grid,
            f_grid=self.f_grid,
            lamb0=self.lamb0,
            ref_idx=self.ref_idx,
            f=self.focal_length,
            NA=self.NA
        )
        
class ThinLens(nn.Module):
    def __init__(self, lens_ref_idx: float, thickness: Tensor, focal_length: float, NA: float):
        super(ThinLens).__init__()
        self.lens_ref_idx = lens_ref_idx
        self.thickness = thickness # Tensro; H, W
        self.focal_length = focal_length
        self.NA = NA
        
    def thick2phase(self, lamb0):
        max_thick = self.thickness.max()
        k = 2 * np.pi / lamb0[None, :, None, None] # 1,C,1,1
        phase1 = max_thick * k  # 1, C, 1, 1
        
        phase2 = k * (self.lens_ref_idx - 1) * self.thickness[None, None]
        
        return torch.exp(1j*phase1) * torch.exp(1j*phase2)
    
    def forward(self, field, grid, lamb0, ref_idx):
        phase = self.thick2phase(lamb0) # e^{i \phi}; 1, C, H, W
        phase_mask(
            field=field,
            phase=phase,
            grid=grid,
            lamb0=lamb0,
            ref_idx=ref_idx,
            f=self.focal_length,
            NA=self.NA
        )
        

## pupil functions (based on https://github.com/chromatix-team/chromatix/blob/main/src/chromatix/functional/pupils.py#L7)
### in small size, circular pupil would not be circularly symmetric
def circular_pupil(grid, d): # d is the diameter
    # grid: 2, H, W
    l2norm_grid = torch.sum(grid ** 2, dim=0) # H, W
    mask = l2norm_grid <= (d/2)**2
    return mask
    
def square_pupil(field, w): # w : width
    mask = torch.max(torch.abs(field), dim=0) <= w/2
    return mask

def multiplicative_phase_lens(field, grid, lamb0, ref_idx, f, NA=None, epsilon=np.finfo(np.float32).eps):
    """
        t_l(x,y) = exp(jk/(2f) * (x^2 + y^2))
    """
    ### lamb : C
    ### NA : 1, C, 1, 1 or float.
    # f is the focal length
    k = 2 * np.pi * ref_idx / lamb0[None, :, None, None] # 1, C, 1, 1
    # grid : Tensor (2, H, W)
    multiplicative_phase = k / (2*f + epsilon) * (torch.sum(grid ** 2, dim=0, keepdim=True).unsqueeze(0)) # B, C, H, W
    
    multp_filter = torch.exp(1j * multiplicative_phase)
    
    if NA is not None:
        D = 2 * f * NA / ref_idx # B, C, 1, 1
        pupil_mask = circular_pupil(grid, D)
        field = field * pupil_mask
    return multp_filter * field


def fflens(field, grid, f_grid, lamb0, ref_idx, f, NA=None):
    if NA is not None:
        D = 2 * f * NA / ref_idx
        pupil_mask = circular_pupil(grid, D) # H, W
        field = field * pupil_mask[None, None ,: ,:]
    return optical_fft(field, grid, f_grid, lamb0, f, ref_idx)
    
def phase_mask(field, phase, grid, ref_idx, f, NA=None):
    
    if NA is not None:
        D = 2 * f * NA / ref_idx
        pupil_mask = circular_pupil(grid, D) # H, W
        field = field * pupil_mask[None, None ,: ,:]
    
    if torch.allclose(phase.real, phase):
        phase = torch.exp(1j*phase) # if phase is angle, then transform the angle into phase.
    
    return field * phase    


