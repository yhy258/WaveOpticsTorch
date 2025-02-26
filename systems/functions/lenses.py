import numpy as np
import scipy.fftpack as fft
import torch


__all__ = ['planesource', 'pointsource']

def compute_intensity(field):
    # field : B, C, H, W -> return : B, 1, H, W
    return torch.sum(torch.abs(field) ** 2, dim=1, keepdim=True)

def compute_power(field, pixel_size):
    pixel_area = pixel_size * pixel_size
    intensity = compute_intensity(field)
    return torch.sum(intensity, dim=(-2, -1), keepdim=True) * pixel_area # B, 1, 1, 1
    
def pointsource(grid, amplitude, lamb0, ref_idx, src_loc, z, power=1.0, epsilon=1e-8):
    # lamb0 : C,
    grid_x, grid_y = grid # H, W
    # src_loc : physical location of point source (center is at the origin)
    # src_loc : B, 2 (different src_loc in batches)
    x, y = src_loc[:, 0], src_loc[:, 1]
    x, y = x[:, None, None], y[:, None, None]
    
    distance = ((grid_x-x)**2 + (grid_y-y) ** 2 + z ** 2) ** (1/2)
    # distance : B, H, W
    
    phase = 2*np.pi*ref_idx/lamb0[None, :, None, None] * (distance[:, None, :, :])
    # lamb0(1 x C x 1 x 1) * distance(B, 1, H, W)
    # field = amplitude * 
    ### https://github.com/chromatix-team/chromatix/blob/main/src/chromatix/functional/sources.py#L67
    field = amplitude * 1/(1j * lamb0*z/ref_idx + epsilon) * torch.exp(1j * phase)
    
    pixel_size = torch.abs(grid_x[1,0] - grid_x[0, 0])
    field_power = compute_power(field, pixel_size)
    return field * torch.sqrt(power / field_power)

### plane source X matter z.
def planesource(grid, amplitude, lamb0, ref_idx, dir_factors, power=1.0):
    """
    If dir_mode == 'dir', then the dir_factors are alpha and betas, where sqrt(alpha ** 2 + beta ** 2 + gamma ** 2) = 1
    If dir_mode == 'f', then the dir_factors are fx and fy (frequency factors), where sqrt(fx**2 + fy**2 + fz**2) = n/lamb0
    If dir_mode == 'k', then the dir_factors are kx and ky (wavevector components), where sqrt(kx**2 + ky**2 + kz**2) = |k|
    """
    # k factors : B, 3 (different k factors in batches)
    # The scale of these factors does not matter.
    grid_x, grid_y = grid # H, W
    
    dir_x, dir_y, dir_z = dir_factors[:, 0], dir_factors[:, 1], dir_factors[:, 2]
    
    # MAKE alpha, beta, and gamma, where l2_norm(dir_factors) = 1
    dir_l2norm = (dir_x**2 + dir_y ** 2+ dir_z ** 2)**(1/2)
    dir_x, dir_y, dir_z = dir_x/dir_l2norm, dir_y/dir_l2norm, dir_z/dir_l2norm
    
    dir_x, dir_y = dir_x[:, None, None, None], dir_y[:, None, None, None] # B, 1, 1, 1
    
    # we do not have to consider the z factor because the phase is just relative...
    inner_dist = (dir_x * grid_x + dir_y * grid_y)
    
    mult_factor = 2 * np.pi * ref_idx / lamb0[None, :, None, None]
    
    phase = mult_factor * inner_dist
    field = amplitude * torch.exp(1j * phase)
    
    pixel_size = torch.abs(grid_x[1,0] - grid_x[0, 0])
    field_power = compute_power(field, pixel_size)
    return field * torch.sqrt(power / field_power)
    
    
    