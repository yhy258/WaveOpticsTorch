import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import math
import numpy as np
import torch
import torch.nn as nn
from systems.systems import Field
from torch.functional import Tensor
from utils import padded_fftnd, unpadded_ifftnd, shifted_fft, double_padnd, double_unpadnd, set_spatial_grid, set_freq_grid

class ASMPropagation(nn.Module):
    def __init__(self, z, ref_idx, band_limited=True):
        super(ASMPropagation, self).__init__()
        self.z = z
        self.ref_idx = ref_idx
        self.band_limited = band_limited
    
    def forward(self, field):
        field.field = double_padnd(field.field, n=2)
        propagator, fx, fy, dfx, dfy = asm_propagator(field, self.z, self.ref_idx) # C, H, W
        
        if self.band_limited:
            propagator = asm_bandlimit(propagator, field.lamb0/self.ref_idx, fx, fy, dfx, dfy, self.z).unsqueeze(0)
        else:
            propagator = propagator.unsqueeze(0)
        
        assert field.field.shape == propagator.shape, f"The input field's shape and the propagator's shape should be same. The input field's shape is {field.shape}, and the propagator's shape is {propagator.shape}"
        return field_propagate(field, propagator, ndim=2)
    
# TODO: THIS DOES NOT WORK.
class SASPropagation(nn.Module):
    def __init__(self, z: float, ref_idx: float, Lx: float, Ly: float):
        super(SASPropagation, self).__init__()
        self.z =z 
        self.ref_idx = ref_idx

        
    def forward(self, field):
        # scalable_ASM(input_field: Tensor, grid: Tensor, f_grid: Tensor, z: float, n: float, lamb0: Tensor, Lx: float, Ly: float)
        output_field, pixel_sizes = scalable_ASM(
            input_field=field,
            z=self.z,
            n=self.ref_idx,
        )
        
        ### define the new grid
        # dx, dy = pixel_sizes
        # H, W = output_field.shape[-2:]
        ### field's grid should be varied
        
        H, W = field.shape[-2:]
        field.x_grid, field.y_grid = set_spatial_grid(H, W, field.dx, field.dy)
        field.fx_grid, field.fy_grid = set_freq_grid(H, W, field.dx, field.dy) # re set.
        return output_field, pixel_sizes


def asm_propagator(field: Field, z: float, n: float):
    # input_field : B, C,.. , H, W, ; complex tensor
    # lamb: Tensor, C,
    # Lx & Ly ; Physical size of image space.
    device = field.device
    H, W = field.shape[-2:]
    
    
    lamb = field.lamb0/n
    
    Lx = field.dx * H
    Ly = field.dy * W
    
    dfx = 1 / Lx
    dfy = 1 / Ly
    
    # d : inverse of sampling rate. sampling rate : 1/Lx
    # define those with frequency instead of wavevector
    fx = torch.fft.fftshift(torch.fft.fftfreq(H, field.dx)).unsqueeze(1).to(device) # H, 1
    fy = torch.fft.fftshift(torch.fft.fftfreq(H, field.dy)).unsqueeze(0).to(device) # 1, W
    
    # lamb : C,
    # fx : H, W
    fz = torch.sqrt((1/lamb[:, None, None]) ** 2 - (fx.unsqueeze(0) **2 + fy.unsqueeze(0)**2)) # broadcasting 사용.
    kz = 2 * np.pi * fz
    propagator = torch.exp(1j * kz * z) # C, H, W
    
    ### BL-ASM ; Antialiasing
    ### Local frequency of phase -> bandlimit
    ### Sampling in Fourier domain -> periodicty in Spatial domain..
    return propagator, fx.unsqueeze(0), fy.unsqueeze(0), dfx, dfy # Inherently, the propagator's DC is at the center of array.

def asm_bandlimit(propagator, lamb, fx, fy, dfx, dfy, z):
    f_limit_px = ((2 * dfx * z)**2 + 1) ** (-1/2) / lamb
    f_limit_nx = -((2 * dfx * z)**2 + 1) ** (-1/2) / lamb
    
    f_limit_py = ((2 * dfy * z)**2 + 1) ** (-1/2) / lamb
    f_limit_ny = -((2 * dfy * z)**2 + 1) ** (-1/2) / lamb
    
    width_x = f_limit_px - f_limit_nx
    width_y = f_limit_py - f_limit_ny
    
    H_filter_x = torch.abs(fx) <= width_x[:, None, None] / 2 # C, H, 1
    H_filter_y = torch.abs(fy) <= width_y[:, None, None] / 2 # C, 1, W
    H_filter = H_filter_x * H_filter_y # C, H, W
    
    return propagator * H_filter
    

def field_propagate(field, propagator, ndim):
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
    field_freq = torch.fft.fftn(field.field, dim=transform_dims)
    
    ### Applying the transfer function
    field_filtered = field_freq * propagator
    
    ### Ifft + unpadding + fftshift
    out = unpadded_ifftnd(field_filtered, n=ndim)
    field.field = out
    ### out's DC is at the center of the array.
    return field
    


### TODO - Scalable ASM requires various dx, dy for different lambda....!!
### Chromaticity...
def scalable_ASM(field: Field, z: float, n: float):
    """
    Args:
        field (Field; B, C, H, W): input field before padding.
        grid (Tensor; 2, H, W): Spatial domain grid
        f_grid (Tensor; 2, H, W): Frequency domain grid
        z (float): 
        n (float): 
        lamb0 (Tensor; C): 
        Lx (float): Physical size of input field
        Ly (float): Physical size of input field
    """
    
    H, W = field.shape[-2:]
    lamb = field.lamb0 / n # C,
    k = 2 * np.pi / lamb # C,
    
    def zbound(L, N):
        return (- 4 * L * np.sqrt(8*L**2 / N**2 + lamb**2) * np.sqrt(L**2 * 1 / (8 * L**2 + N**2 * lamb**2))\
               / (lamb * (-1+2 * np.sqrt(2) * np.sqrt(L**2 * 1 / (8 * L**2 + N**2 * lamb**2)))))
        
    ### Bound of z.
    z_limit_x = zbound(field.Lx, H)
    z_limit_y = zbound(field.Ly, W)
    assert torch.allclose((z <= z_limit_x).type(torch.int) * (z <= z_limit_y).type(torch.int), torch.ones_like(z_limit_x, dtype=torch.int)), f"The provided z value goes beyond the distance limit. {min(z_limit_x, z_limit_y)}"

    # define new grids (double padded)
    
    newLx, newLy = (2 * H - 1) * field.dx, (2 * W - 1) * field.dy
    newH, newW = (2 * H - 1), (2 * W - 1)

    ### 1. new spatial grid
    grid = set_spatial_grid(newH, newW, field.dx, field.dy)
    
    ### 2. new freq grid
    f_grid = set_freq_grid(newH, newW, field.dx, field.dy)
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
    field.field = double_padnd(field.field, n=2) ### DC at starting point of the array and double padded.
    
    ### Apply compensation kernel
    field.field = torch.fft.ifft2(torch.fft.fft2(field.field) * comp_kernel.unsqueeze(0)) ### DC at starting oint of the array.
    
    ### SFT-FR
    output_field, new_pixel_sizes = fourier_repr_fresnel_prop(field, grid, z, n, newLx, newLy)
    ### unpad..
    output_field = double_unpadnd(output_field)
    
    return output_field, new_pixel_sizes # with this value, we can derive the new grid, because the number of pixels are preserved.
    
    
    

def fourier_repr_fresnel_prop(field: Field, grid: Tensor, z: float, n: float, Lx: float, Ly: float):
    ### INPUT; (B, C, H, W) -> (B, C, H, W) [same shape], but the grid step is differet (magnification)
    """
    Args:
        field (Field; B, C, H, W): Input field. - double padded.
        grid (Tensor; 2, H, W): double padded grid.
        z (float): The propagation distance
        n (float): refractive index along propagation space
        lamb0 (Tensor; C,): wavelengths
        Lx (float): Physical size of input field
        Ly (float): Physical size of input field
    """
    
    H, W = field.shape[-2:]
    k = 2*np.pi*n/field.lamb0 # C,
    lamb = field.lamb0 / n
    
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
    Q2_front = torch.exp(1j*k[:, None, None]*z)/(1j * field.lamb0[:, None, None]/n * z)
    Q2_end = torch.exp(1j * k[:, None, None] / (2*z) * (dest_grid_x ** 2 + dest_grid_y ** 2))  # C, H, W
    Q2 = Q2_front * Q2_end
    
    # 1. Q1 * input field
    # Both Q1 and input field are at center instead of left-upper corner. - These Q1 and input field are regarded as Foureir components.
    return Q2*shifted_fft(field * Q1.unsqueeze(0)), (dest_dx, dest_dy) # shifted_fft(input_field * Q1)'s DC component would be at the center of the array.
