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
    
# DSF; Double Step Fresnel Diffraction
# Usually z1, z2 ; z/2 + 500(m), z/2 - 500(m)
# TODO: including band_limited version
class DSFPropagation(nn.Module):
    def __init__(self, z, ref_idx, band_limited=True):
        super(DSFPropagation, self).__init__()
        self.z = z
        self.ref_idx = ref_idx
        self.band_limited = band_limited
    
    # The default values of the propagation is the basic setting of the given paper.
    # If setting source_parabolic and target_parabolic as False, then it is equal to Fraunhofer Diffraction..
    def forward(self, field: Field):
        z1 = self.z / 2 + 500 * 1e6
        z2 = self.z / 2 - 500 * 1e6
        field =  double_step_fresnel_diffraction(field, z1, z2, self.ref_idx, band_limit=self.band_limited)
        return field
    
# TODO: THIS DOES NOT WORK.
# class SASPropagation(nn.Module):
#     def __init__(self, z: float, ref_idx: float, Lx: float, Ly: float):
#         super(SASPropagation, self).__init__()
#         self.z =z 
#         self.ref_idx = ref_idx

        
#     def forward(self, field):
#         # scalable_ASM(input_field: Tensor, grid: Tensor, f_grid: Tensor, z: float, n: float, lamb0: Tensor, Lx: float, Ly: float)
#         output_field, pixel_sizes = scalable_ASM(
#             input_field=field,
#             z=self.z,
#             n=self.ref_idx,
#         )
        
#         ### define the new grid
#         # dx, dy = pixel_sizes
#         # H, W = output_field.shape[-2:]
#         ### field's grid should be varied
        
#         H, W = field.shape[-2:]
#         field.x_grid, field.y_grid = set_spatial_grid(H, W, field.dx, field.dy)
#         field.fx_grid, field.fy_grid = set_freq_grid(H, W, field.dx, field.dy) # re set.
#         return output_field, pixel_sizes


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
    


# ### TODO - Scalable ASM requires various dx, dy for different lambda....!!
# ### Chromaticity...
# def scalable_ASM(field: Field, z: float, n: float):
#     """
#     Args:
#         field (Field; B, C, H, W): input field before padding.
#         grid (Tensor; 2, H, W): Spatial domain grid
#         f_grid (Tensor; 2, H, W): Frequency domain grid
#         z (float): 
#         n (float): 
#         lamb0 (Tensor; C): 
#         Lx (float): Physical size of input field
#         Ly (float): Physical size of input field
#     """
    
#     H, W = field.shape[-2:]
#     lamb = field.lamb0 / n # C,
#     k = 2 * np.pi / lamb # C,
    
#     def zbound(L, N):
#         return (- 4 * L * np.sqrt(8*L**2 / N**2 + lamb**2) * np.sqrt(L**2 * 1 / (8 * L**2 + N**2 * lamb**2))\
#                / (lamb * (-1+2 * np.sqrt(2) * np.sqrt(L**2 * 1 / (8 * L**2 + N**2 * lamb**2)))))
        
#     ### Bound of z.
#     z_limit_x = zbound(field.Lx, H)
#     z_limit_y = zbound(field.Ly, W)
#     assert torch.allclose((z <= z_limit_x).type(torch.int) * (z <= z_limit_y).type(torch.int), torch.ones_like(z_limit_x, dtype=torch.int)), f"The provided z value goes beyond the distance limit. {min(z_limit_x, z_limit_y)}"

#     # define new grids (double padded)
    
#     newLx, newLy = (2 * H - 1) * field.dx, (2 * W - 1) * field.dy
#     newH, newW = (2 * H - 1), (2 * W - 1)

#     ### 1. new spatial grid
#     grid = set_spatial_grid(newH, newW, field.dx, field.dy)
    
#     ### 2. new freq grid
#     f_grid = set_freq_grid(newH, newW, field.dx, field.dy)
#     fx, fy = f_grid[0], f_grid[1]
    
#     ### bandlimit helper
#     cx = lamb[:, None, None] * fx.unsqueeze(0)
#     cy = lamb[:, None, None] * fy.unsqueeze(0)
#     tx = newLx / 2 / z + torch.abs(lamb[:, None, None] * fx.unsqueeze(0))
#     ty = newLy / 2 / z + torch.abs(lamb[:, None, None] * fy.unsqueeze(0))
    
#     ### bandlimit filter for precompensation. ; C, H, W
#     W = (cx**2 * (1 + tx**2) / tx**2 + cy**2 <= 1) * (cy**2 * (1 + ty**2) / ty**2 + cx**2 <= 1)
    
#     # calculate kernels
#     lfx, lfy = lamb[:, None, None] * fx.unsqueeze(0), lamb[:, None, None] * fy.unsqueeze(0)
#     asm_kernel_inner_phase = torch.sqrt(1 - lfx ** 2 - lfy**2) # C, H, W
#     fresnel_kernel_inner_phase = 1 - lfx**2/2 - lfy**2/2 # C, H, W
#     comp_kernel = torch.fft.ifftshift(torch.exp(1j*k[:, None, None]*z*(asm_kernel_inner_phase - fresnel_kernel_inner_phase)), dim=(-2, -1)) # DC at starting point of the array.

#     ### padding
#     field.field = double_padnd(field.field, n=2) ### DC at starting point of the array and double padded.
    
#     ### Apply compensation kernel
#     field.field = torch.fft.ifft2(torch.fft.fft2(field.field) * comp_kernel.unsqueeze(0)) ### DC at starting oint of the array.
    
#     ### SFT-FR
#     output_field, new_pixel_sizes = fourier_repr_fresnel_prop(field, grid, z, n, newLx, newLy)
#     ### unpad..
#     output_field = double_unpadnd(output_field)
    
#     return output_field, new_pixel_sizes # with this value, we can derive the new grid, because the number of pixels are preserved.
    
    
    

# def fourier_repr_fresnel_prop(field: Field, grid: Tensor, z: float, n: float, Lx: float, Ly: float):
#     ### INPUT; (B, C, H, W) -> (B, C, H, W) [same shape], but the grid step is differet (magnification)
#     """
#     Args:
#         field (Field; B, C, H, W): Input field. - double padded.
#         grid (Tensor; 2, H, W): double padded grid.
#         z (float): The propagation distance
#         n (float): refractive index along propagation space
#         lamb0 (Tensor; C,): wavelengths
#         Lx (float): Physical size of input field
#         Ly (float): Physical size of input field
#     """
    
#     H, W = field.shape[-2:]
#     k = 2*np.pi*n/field.lamb0 # C,
#     lamb = field.lamb0 / n
#     dx_denom = lamb * z
    
#     # Input field's coordinate
#     src_dx, src_dy = field.dx, field.dy
    
#     # Destination coordinate
#     dest_dx, dest_dy = src_dx / dx_denom, src_dy / dx_denom
    
#     # magnification factors; C,
#     mag_x, mag_y = dest_dx / src_dx, dest_dy / src_dy
    
#     # src_grid : H, 1 and 1, W
#     src_grid_x, src_grid_y = field.x_grid, field.y_grid
    
#     # dest_grid : 
#     field.set_grid(dest_dx, dest_dy)
#     field.set_fgrid(dest_dx, dest_dy)
#     dest_grid_x, dest_grid_y = field.x_grid[0], field.y_grid[1]
#     # Destination grid; C, H, W
#     dest_grid_x, dest_grid_y = src_grid_x.unsqueeze(0) * mag_x[:, None, None], src_grid_y.unsqueeze(0) * mag_y[:, None, None]
    
#     ### ASSUME : ENTIRE INPUTS ARE AT CENTER. THAT MEANS WHEN WE CONDUCT FOURIER TRANSFORM, WE HAVE TO FUNCTION IFFTSHIFT.
#     ### The reason why the above assumption is satisfied is that the inputs would be Fourier-transformed. This input-field have to be considered in Fourier domain.
#     Q1 = torch.exp(1j * k[:, None, None] / (2*z) * (src_grid_x.unsqueeze(0) ** 2 + src_grid_y.unsqueeze(0) ** 2))  # C, H, W
#     Q2_front = torch.exp(1j*k[:, None, None]*z)/(1j * field.lamb0[:, None, None]/n * z)
#     Q2_end = torch.exp(1j * k[:, None, None] / (2*z) * (dest_grid_x ** 2 + dest_grid_y ** 2))  # C, H, W
#     Q2 = Q2_front * Q2_end
    
#     # 1. Q1 * input field
#     # Both Q1 and input field are at center instead of left-upper corner. - These Q1 and input field are regarded as Foureir components.
#     dest_field = Q2*shifted_fft(field * Q1.unsqueeze(0))
#     ### Re-define the grid.
#     dest_field.set_grid(dest_dx, dest_dy)
#     dest_field.set_fgrid(dest_dx, dest_dy)
    

"""
    In this framework, we have to consider the several wavelengths.
    The Fourier transform representation of fresnel diffraction (single step fresnel diffraction; SSF) changes the grid information.
    (fx = x /(lamb * z))
    So, in order to enable this functional, we should revise the entire code, making the whole codes' grid 3D (C, H, W)
    Additionally, SSF method outcomes the various grid sizes for several wavelengths because of the above reason.
    Consequently, we decided to neglect the grid changes. We just use the SSF to perform the double step fresnel diffraction.
"""
def single_step_fresnel_diffraction(field: Field, x_grid: Tensor, y_grid: Tensor, z: float, n: float, target_parabolic: bool = True, source_parabolic: bool = True):
    """
    Args:
        field (Field): Input field with shape (B, C, H, W)
        x_grid & y_grid (Tensor): Input field's grid (C, H, W)
        z (float): Propagation distance
        n (float): Refractive Index
        target_parabolic (bool, optional): Whether applying the parabolic term for the destination grid.
        source_parabolic (bool, optional): Whether applying the parabolic term for the source grid.
    """
    _, _, H, W =  field.shape
    k = 2*np.pi*n/field.lamb0[:, None, None] # C, 1, 1
    lamb = field.lamb0[:, None, None] / n # C, 1, 1
    dx_denom = lamb * z # C, 1, 1
    
    # Input field's coordinate
    # Don't use the Field Object's grid;
    if len(x_grid.shape) < 3 or len(y_grid.shape) < 3:
        src_grid_x, src_grid_y = x_grid.unsqueeze(0), y_grid.unsqueeze(0) # 1, H, 1 and 1, 1, W
    else:
        src_grid_x, src_grid_y = x_grid, y_grid # 1, H, 1 and 1, 1, W
    dx, dy = torch.abs(src_grid_x[:,1,0] - src_grid_x[:,0,0]), torch.abs(src_grid_y[:,0,1] - src_grid_y[:,0,0])

    # fx = x_2 / (lamb * z), fy = y_2 / (lamb * z)
    # x_2, y_2 are the fourier grid * lamb * z
    # dest_grid_x, dest_grid_y = set_spatial_grid(H, W, dest_dx, dest_dy) # This is same with set_freq_grid(H,W, dx=field.dx, dy=field.dy) * lamb * z
    # dx, dy : C, 
    # I have to improve the definition code of the frequency grid.
    new_grid_xs = []
    new_grid_ys = []
    for dx_, dy_ in zip(dx, dy):
        new_grid_x, new_grid_y = set_freq_grid(H, W, dx_, dy_)
        new_grid_xs.append(new_grid_x)
        new_grid_ys.append(new_grid_y)
    dest_grid_x = torch.stack(new_grid_xs).to(field.device)
    dest_grid_y = torch.stack(new_grid_ys).to(field.device)
    final_coef = torch.exp(1j*k*z) / (1j * lamb * z) # C, 1, 1
    
    ### field : B, C, H, W
    if source_parabolic:
        Q1 = torch.exp(1j * k / (2*z) * (src_grid_x ** 2 + src_grid_y ** 2)).unsqueeze(0) # 1, C, H, W
    else:
        Q1 = 1.
        
    if target_parabolic:
        Q2 = torch.exp(1j * (k / (2*z)) * (dest_grid_x ** 2 + dest_grid_y ** 2)).unsqueeze(0) # 1, C, H, W
    else:
        Q2 = 1.
    
    
    inv = True if z < 0 else False
    dest_field = shifted_fft(field * Q1, inv=inv) * Q2 * final_coef # B, C, H, W & with x/(lambda * z)
    return dest_field, dest_grid_x, dest_grid_y
    
    


# Band-limited double-step Fresnel diffraction and its application to computer-generated holograms
# https://opg.optica.org/oe/fulltext.cfm?uri=oe-21-7-9192&id=252408
# Usually z1, z2 ; z/2 + 500(m), z/2 - 500(m)
# def double_step_fresnel_diffraction(field: Field, z1: float, z2: float, n: float, target_parabolic: bool = True, source_parabolic: bool = True):
#     """
#     Args:
#         field (Field): Input field with shape (B, C, H, W)
#         z1 (float): The first propagation distance
#         z2 (float): The second propagation distance # z1 + z2 = z
#         n (float): Refractive Index
#         target_parabolic (bool, optional): Whether applying the parabolic term for the destination grid.
#         source_parabolic (bool, optional): Whether applying the parabolic term for the source grid.
#     """
    
    
#     field1, grid_x, grid_y = single_step_fresnel_diffraction(field, field.x_grid, field.y_grid, z1, n, target_parabolic=target_parabolic, source_parabolic=source_parabolic)
#     return single_step_fresnel_diffraction(field1, grid_x, grid_y, z2, n, target_parabolic=target_parabolic, source_parabolic=source_parabolic)


#### 
def double_step_fresnel_diffraction(field: Field, z1: float, z2: float, n: float, band_limit: bool = True):
    _, _, H, W =  field.shape
    k = 2*np.pi*n/field.lamb0[:, None, None] # C, 1, 1
    lamb = field.lamb0[:, None, None] / n # C, 1, 1
    # initial grid info
    dx1, dx2 = field.dx, field.dy
    
    ##### first propagation
    Q1 = torch.exp(1j * k / (2*z1) * (field.x_grid ** 2 + field.y_grid ** 2)).unsqueeze(0) # 1, C, H, W

        
    # final_coef = torch.exp(1j*k*z1) / (1j * lamb * z1) # C, 1, 1
    inv = True if z1 < 0 else False
    fieldv = shifted_fft(field * Q1, inv=inv) # B, C, H, W & with x/(lambda * z)
    
    dxv, dyv = lamb * z1 / field.Lx, lamb * z1 / field.Ly # C, 1, 1
    
    grid_xv, grid_yv = set_spatial_grid(H, W, 1, 1) 
    grid_xv, grid_yv = grid_xv.unsqueeze(0).to(field.device) * dxv, grid_yv.unsqueeze(0).to(field.device) * dyv

    ##### Chirp like function
    chirp_func = torch.exp(1j*np.pi*(z1+z2)*(grid_xv**2 + grid_yv**2)/(lamb * z1*z2)).unsqueeze(0) # 1, C, H, W
    
    ##### band limit
    if band_limit:
        xv_max = torch.abs(lamb*z1*z2)/(dxv * 2 * (z1 + z2))/2
        yv_max = torch.abs(lamb*z1*z2)/(dyv * 2 * (z1 + z2))/2
        xv_mask, yv_mask = torch.abs(grid_xv) <= xv_max, torch.abs(grid_yv) <= yv_max
        mask_filter = (xv_mask * yv_mask).unsqueeze(0)
        chirp_func *= mask_filter
    
    
    final_coef = torch.exp(1j*k*z2) / (1j * lamb * z2) # C, 1, 1
    inv = True if z2 < 0 else False
    
    field = shifted_fft(fieldv * chirp_func, inv=inv) * final_coef.unsqueeze(0) # B, C, H, W & with x/(lambda * z)
    return field