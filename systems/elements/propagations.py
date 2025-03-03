import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)

import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
from utils import padded_fftnd, unpadded_ifftnd, shifted_fft, double_padnd, double_unpadnd, set_spatial_grid, set_freq_grid

"""
    1. Padding the field to alleviate the circular convolution problem.
    2. Define the f-grid using the padded field.
"""
class ASMPropagation(nn.Module):
    def __init__(self, z, lamb0, ref_idx, dx, dy, band_limited=True):
        super(ASMPropagation, self).__init__()
        self.z = z
        self.lamb0 = lamb0
        self.ref_idx = ref_idx
        self.dx = dx
        self.dy = dy
        self.band_limited = band_limited
        
    def forward(self, field):
        return asm_propagation(
            input_field=field,
            z=self.z,
            n=self.ref_idx,
            lamb0=self.lamb0,
            dx=self.dx,
            dy=self.dy,
            band_limited=self.band_limited
        )



# TODO: THIS DOES NOT WORK.
class SASPropagation(nn.Module):
    def __init__(self, grid: Tensor, z: float, lamb0: Tensor, ref_idx: float, Lx: float, Ly: float):
        super(SASPropagation, self).__init__()
        self.grid = grid
        self.z =z 
        self.ref_idx = ref_idx
        self.lamb0 = lamb0
        self.Lx = Lx
        self.Ly = Ly
        
    def forward(self, field):
        # scalable_ASM(input_field: Tensor, grid: Tensor, f_grid: Tensor, z: float, n: float, lamb0: Tensor, Lx: float, Ly: float)
        output_field, pixel_sizes = scalable_ASM(
            input_field=field,
            grid=self.grid,
            z=self.z,
            n=self.ref_idx,
            lamb0=self.lamb0,
            Lx=self.Lx,
            Ly=self.Ly
        )
        
        ### define the new grid
        # dx, dy = pixel_sizes
        # H, W = output_field.shape[-2:]
        return output_field, pixel_sizes


def asm_propagation(input_field, z, n: float, lamb0: Tensor, dx: float, dy: float, band_limited: bool = True):
    input_field = double_padnd(input_field, n=2)
    propagator = asm_propagator(input_field, z, n, lamb0, dx, dy, band_limited)
    return field_propagate(input_field, propagator, ndim=2) # this function includes padding operations to alleviate circular convolution.

def asm_propagator(input_field, z, n: float, lamb0: Tensor, dx: float, dy: float, band_limited: bool = True):
    H, W = input_field.shape[-2:]
    transform_dims = (-np.arange(1, 2+1)[::-1]).tolist()
    # input_field : B, C,.. , H, W, ; complex tensor
    # lamb: Tensor, C,
    # Lx & Ly ; Physical size of image space.
    
    k = 2*np.pi*n / lamb0
    lamb = lamb0/n
    
    Lx = dx * H
    Ly = dy * W
    
    dfx = 1 / Lx
    dfy = 1 / Ly
    
    # d : inverse of sampling rate. sampling rate : 1/Lx
    # define those with frequency instead of wavevector
    fx = np.fft.fftshift(np.fft.fftfreq(H, Lx/H))
    fy = np.fft.fftshift(np.fft.fftfreq(W, Ly/W)) #### center position
    fx, fy = np.meshgrid(fx, fy, indexing='ij')
    fx, fy = torch.from_numpy(fx), torch.from_numpy(fy)
    
    # lamb : C,
    # fx : H, W
    fz = torch.sqrt((1/lamb[:, None, None]) ** 2 - (fx.unsqueeze(0)**2 + fy.unsqueeze(0)**2))
    kz = 2 * np.pi * fz
    propagator = torch.exp(1j * kz * z) # C, H, W
    
    ### BL-ASM ; Antialiasing
    ### Local frequency of phase -> bandlimit
    ### Sampling in Fourier domain -> periodicty in Spatial domain..
    if band_limited:
        # C,
        f_limit_px = ((2 * dfx * z)**2 + 1) ** (-1/2) / lamb
        f_limit_nx = -((2 * dfx * z)**2 + 1) ** (-1/2) / lamb
        
        f_limit_py = ((2 * dfy * z)**2 + 1) ** (-1/2) / lamb
        f_limit_ny = -((2 * dfy * z)**2 + 1) ** (-1/2) / lamb
        
        width_x = f_limit_px - f_limit_nx
        width_y = f_limit_py - f_limit_ny
        
        H_filter_x = torch.abs(fx).unsqueeze(0) <= width_x[:, None, None]
        H_filter_y = torch.abs(fy).unsqueeze(0) <= width_y[:, None, None]
        H_filter = H_filter_x * H_filter_y # C, H, W
        
        propagator = propagator * H_filter
    propagator = propagator.unsqueeze(0) # C, H, W -> 1, C, H, W
    return propagator # Inherently, the propagator's DC is at the center of array.

def field_propagate(input_field, propagator, ndim):
    # input_field : B, C, ..., H, W
    # propagator : 1, 1, H, W
    
    transform_dims = (-np.arange(1, ndim+1)[::-1]).tolist()
    
    for n in range(ndim-2):
        propagator = propagator.unsqueeze(0)
    
    #### The frequency mesh's DC of the propagator is at the center of the array..
    ### In order to appropriately conduct pytorch's fft, we have to relocate the position of DC components to starting point of array.
    #### This can be performed by utilizing ifftshift.
    
    # the DC component of propagator at the center of the array
    propagator = torch.fft.ifftshift(propagator, dim=transform_dims)
    
    ### fft.
    field_freq = torch.fft.fftn(input_field, dim=transform_dims)
    
    ### Applying the transfer function
    field_filtered = field_freq * propagator
    
    ### Ifft + unpadding + fftshift
    out = unpadded_ifftnd(field_filtered, n=ndim)
    ### out's DC is at the center of the array.
    return out
    


### TODO - Scalable ASM requires various dx, dy for different lambda....!!
### Chromaticity...
def scalable_ASM(input_field: Tensor, grid: Tensor, z: float, n: float, lamb0: Tensor, Lx: float, Ly: float):
    """
    Args:
        input_field (Tensor; B, C, H, W): input field before padding.
        grid (Tensor; 2, H, W): Spatial domain grid
        f_grid (Tensor; 2, H, W): Frequency domain grid
        z (float): 
        n (float): 
        lamb0 (Tensor; C): 
        Lx (float): Physical size of input field
        Ly (float): Physical size of input field
    """
    
    H, W = input_field.shape[-2:]
    lamb = lamb0 / n # C,
    k = 2 * np.pi / lamb # C,
    
    def zbound(L, N):
        return (- 4 * L * np.sqrt(8*L**2 / N**2 + lamb**2) * np.sqrt(L**2 * 1 / (8 * L**2 + N**2 * lamb**2))\
               / (lamb * (-1+2 * np.sqrt(2) * np.sqrt(L**2 * 1 / (8 * L**2 + N**2 * lamb**2)))))
        
    ### Bound of z.
    z_limit_x = zbound(Lx, H)
    z_limit_y = zbound(Ly, W)
    assert torch.allclose((z <= z_limit_x).type(torch.int) * (z <= z_limit_y).type(torch.int), torch.ones_like(z_limit_x, dtype=torch.int)), f"The provided z value goes beyond the distance limit. {min(z_limit_x, z_limit_y)}"

    # define new grids (double padded)
    dx, dy = torch.abs(grid[0, 0, 0] - grid[0, 1, 0]), torch.abs(grid[1, 0, 0] - grid[1, 0, 1])
    newLx, newLy = (2 * H - 1) * dx, (2 * W - 1) * dy
    newH, newW = (2 * H - 1), (2 * W - 1)

    ### 1. new spatial grid
    grid = set_spatial_grid(newH, newW, dx, dy)
    
    ### 2. new freq grid
    f_grid = set_freq_grid(newH, newW, dx.item(), dy.item())
    fx, fy = f_grid[0], f_grid[1]
    
    ### bandlimit helper
    cx = lamb[:, None, None] * fx.unsqueeze(0)
    cy = lamb[:, None, None] * fy.unsqueeze(0)
    tx = newLx / 2 / z + torch.abs(lamb[:, None, None] * fx.unsqueeze(0))
    ty = newLy / 2 / z + torch.abs(lamb[:, None, None] * fy.unsqueeze(0))
    
    ### bandlimit filter for precompensation. ; C, H, W
    W = (cx**2 * (1 + tx**2) / tx**2 + cy**2 <= 1) * (cy**2 * (1 + ty**2) / ty**2 + cx**2 <= 1)
    
    # calculate kernels
    lfx, lfy = lamb[:, None, None] * fx.unsqueeze(0), lamb[:, None, None] * fy.unsqueeze(0)
    asm_kernel_inner_phase = torch.sqrt(1 - lfx ** 2 - lfy**2) # C, H, W
    fresnel_kernel_inner_phase = 1 - lfx**2/2 - lfy**2/2 # C, H, W
    comp_kernel = torch.fft.ifftshift(torch.exp(1j*k[:, None, None]*z*(asm_kernel_inner_phase - fresnel_kernel_inner_phase)), dim=(-2, -1)) # DC at starting point of the array.

    ### padding
    input_field = double_padnd(input_field, n=2) ### DC at starting point of the array and double padded.
    
    ### Apply compensation kernel
    input_field = torch.fft.ifft2(torch.fft.fft2(input_field) * comp_kernel.unsqueeze(0)) ### DC at starting oint of the array.
    
    ### SFT-FR
    output_field, new_pixel_sizes = fourier_repr_fresnel_prop(input_field, grid, z, n, lamb0, newLx, newLy)
    ### unpad..
    output_field = double_unpadnd(output_field)
    
    return output_field, new_pixel_sizes # with this value, we can derive the new grid, because the number of pixels are preserved.
    
    
    

def fourier_repr_fresnel_prop(input_field: Tensor, grid: Tensor, z: float, n: float, lamb0: Tensor, Lx: float, Ly: float):
    ### INPUT; (B, C, H, W) -> (B, C, H, W) [same shape], but the grid step is differet (magnification)
    """
    Args:
        input_field (Tensor; B, C, H, W): Input field. - double padded.
        grid (Tensor; 2, H, W): Input field's grid.
        z (float): The propagation distance
        n (float): refractive index along propagation space
        lamb0 (Tensor; C,): wavelengths
        Lx (float): Physical size of input field
        Ly (float): Physical size of input field
    """
    
    H, W = input_field.shape[-2:]
    k = 2*np.pi*n/lamb0 # C,
    lamb = lamb0 / n
    
    # Input field's coordinate
    src_dx, src_dy = Lx / H, Ly / W
    
    # Destination coordinate
    dest_dx, dest_dy = lamb * z / Lx, lamb * z / Ly 
    
    # magnification factors; C,
    mag_x, mag_y = dest_dx / src_dx, dest_dy / src_dy
    
    # src_grid : H, W
    src_grid_x, src_grid_y = grid[0], grid[1]
    
    # Destination grid; C, H, W
    dest_grid_x, dest_grid_y = src_grid_x.unsqueeze(0) * mag_x[:, None, None], src_grid_y.unsqueeze(0) * mag_y[:, None, None]
    
    ### ASSUME : ENTIRE INPUTS ARE AT CENTER. THAT MEANS WHEN WE CONDUCT FOURIER TRANSFORM, WE HAVE TO FUNCTION IFFTSHIFT.
    ### The reason why the above assumption is satisfied is that the inputs would be Fourier-transformed. This input-field have to be considered in Fourier domain.
    Q1 = torch.exp(1j * k[:, None, None] / (2*z) * (src_grid_x.unsqueeze(0) ** 2 + src_grid_y.unsqueeze(0) ** 2))  # C, H, W
    Q2_front = torch.exp(1j*k[:, None, None]*z)/(1j * lamb0[:, None, None]/n * z)
    Q2_end = torch.exp(1j * k[:, None, None] / (2*z) * (dest_grid_x ** 2 + dest_grid_y ** 2))  # C, H, W
    Q2 = Q2_front * Q2_end
    
    # 1. Q1 * input field
    # Both Q1 and input field are at center instead of left-upper corner. - These Q1 and input field are regarded as Foureir components.
    return Q2*shifted_fft(input_field * Q1.unsqueeze(0)), (dest_dx, dest_dy) # shifted_fft(input_field * Q1)'s DC component would be at the center of the array.
