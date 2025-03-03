import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
from utils import compute_power

"""
    1. Point source
    2. Plane wave
"""

class PointSource(nn.Module):
    """
        Unlike other classes, Source classes get "grid" and "lamb0" parameters when initializing this instance.
        This is because "field" is derived from the "source".
        That means, the fields' propoerties come from "source definition"
    """
    def __init__(self, grid, amplitude, lamb0, ref_idx, src_loc, z, power=1.0, paraxial=False, epsilon=np.finfo(np.float32).eps):
        super(PointSource, self).__init__()
        self.grid = grid
        self.amplitude = amplitude
        self.lamb0 = lamb0
        self.ref_idx = ref_idx
        self.z = z
        self.src_loc = src_loc
        self.power = power
        self.paraxial = paraxial
        self.epsilon = epsilon
        
    
    def forward(self):
        return pointsource(
            grid=self.grid,
            amplitude=self.amplitude,
            lamb0=self.lamb0,
            ref_idx=self.ref_idx,
            src_loc=self.src_loc,
            z=self.z,
            power=self.power,
            paraxial=self.paraxial
        )
        
        
class PlaneSource(nn.Module):
    
    def __init__(self, grid: Tensor, amplitude: float, lamb0: Tensor, ref_idx: float, power: float, dir_factors: Tensor):
        """
        Args:
            grid (Tensor; 2, H, W): Grid of field..
            amplitude (float): Amplitude of source
            lamb0 (Tensor; C,): Wavelengths
            dir_factors (Tensor; 3,): Direction factors (2pi * n /lambda)(dir_x, dir_y, dir_z).
                                    : BUT, kx, ky, kz or fx, fy, fz are also okay, because I will normalize the factors.
        """
        super(PlaneSource, self).__init__()
        self.grid = grid
        self.amplitude = amplitude
        self.lamb0 = lamb0
        self.ref_idx = ref_idx
        self.dir_factors = dir_factors
        self.power = power
        
    def forward(self):
        return planesource(
            grid=self.grid,
            amplitude=self.amplitude,
            lamb0=self.lamb0,
            ref_idx=self.ref_idx,
            dir_factors=self.dir_factors,
            power=self.power
        )
        

def pointsource(grid, amplitude, lamb0, ref_idx, src_loc, z, power=1.0, paraxial=False, epsilon=np.finfo(np.float32).eps):
    # lamb0 : C,
    if paraxial:
        assert src_loc == None or src_loc == 0 or torch.allclose(src_loc, torch.zeros_like(src_loc)), "In paraxial setting, please set the src_loc = 0 or None"
    
    # grid_x, grid_y = grid[0].unsqueeze(0), grid[1].unsqueeze(0) # 1, H, W
    grid_x, grid_y = grid[0], grid[1] # H, W
    radial_grid = torch.sum(grid ** 2, dim=0) # H, W
    
    # src_loc : physical location of point source (center is at the origin)
    # src_loc : 2 (different src_loc)
    if src_loc == None:
        x, y = 0, 0
    else:
        x, y = src_loc[0], src_loc[1] # floats
        
    k = 2 * np.pi * ref_idx / lamb0[:, None, None] # C, 1, 1
    if paraxial:
        phase_distance = 1/(2*z + epsilon)*radial_grid
        denom_distance = z + epsilon
    else:
        phase_distance = ((grid_x-x)**2 + (grid_y-y) ** 2 + z ** 2) ** (1/2)
        denom_distance = (((grid_x-x)**2 + (grid_y-y) ** 2 + z ** 2) ** (1/2) + epsilon)
        # distance : 1, H, W


    phase = k * (phase_distance.unsqueeze(0)) # 1, C,  H, W
    ### https://github.com/chromatix-team/chromatix/blob/main/src/chromatix/functional/sources.py#L67
    field = amplitude * 1/denom_distance * torch.exp(1j * phase)
    
    
    pixel_area = torch.abs(grid_x[1, 0] - grid_x[0, 0]) * torch.abs(grid_y[0, 1] - grid_y[0, 0])
    field_power = compute_power(field, pixel_area)
    powered_field = field * torch.sqrt(power / field_power)
    
    # 1, C, H, W
    return field.unsqueeze(0), powered_field.unsqueeze(0)




### plane source X matter z.
def planesource(grid, amplitude, lamb0, ref_idx, dir_factors, power=1.0):
    # k factors : 3, (different k factors)
    # The scale of these factors does not matter.
    grid_x, grid_y = grid[0], grid[1] # H, W
    
    # dir_factors : 3
    # dir_x, dir_y, dir_z are just float.
    dir_x, dir_y, dir_z = dir_factors[0], dir_factors[1], dir_factors[2]
    
    # MAKE alpha, beta, and gamma, where l2_norm(dir_factors) = 1
    dir_l2norm = (dir_x**2 + dir_y ** 2+ dir_z ** 2)**(1/2)
    if dir_l2norm != 0:
        dir_x, dir_y, dir_z = dir_x/dir_l2norm, dir_y/dir_l2norm, dir_z/dir_l2norm
    else:
        dir_x, dir_y, dir_z = 0, 0, 1
    
    # dir_x, dir_y = dir_x[:, None, None, None], dir_y[:, None, None, None] # B, 1, 1, 1
    
    # we do not have to consider the z factor because the phase is just relative...
    inner_dist = (dir_x * grid_x[None, None, :, :] + dir_y * grid_y[None, None, :, :])
    
    mult_factor = 2 * np.pi * ref_idx / lamb0[None, :, None, None]
    
    phase = mult_factor * inner_dist
    field = amplitude * torch.exp(1j * phase)
    
    pixel_size = torch.abs(grid_x[1,0] - grid_x[0, 0])
    field_power = compute_power(field, pixel_size)
    powered_field = field * torch.sqrt(power / field_power)
    
    return field, powered_field
