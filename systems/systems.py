"""
    COHERENT vs INCOHERENT
    1. COHERENT ILLUMINATION : ATF (the relationship between output amplitude and input ampliutde is linear with amplitude transfer function.)
        F{AMP_OUTPUT} : ATF*F{AMP_INPUT}
        |AMP_OUTPUT|^2 is detecteed on sensor.
    2. INCOHERENT ILLUMINATION : OTF (the relationship between input intensity and output intensity is linear with optical transfer function; auto-correlation of ATF)
        F{INT_OUTPUT} = OTF * F{INT_INPUT}
        INT_OUTPUT is detected on sensor.
"""

import os, sys
import warnings

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import set_spatial_grid, set_freq_grid, _list, compute_power, compute_intensity

############################## INTEGRATED OPTICAL SYSTEM.
class OpticalSystem(nn.Module):
    # unit : µm
    # still do not care "magnification"
    meter = 1e+6
    micrometers = 1
    nanometers = 1e-3
    
    def __init__(
        self,
        pixel_size: list = [5, 5],
        pixel_num: list = [1000, 1000],
        ### noise parameters
        poisson_m: float = 0.,
        gaussian_std: float = 1.,
        ### wavelength in free space
        lamb0: list = [], # chromaticity, meter.
        refractive_index: float = 1., # Air = 1.
    ):
        super(OpticalSystem, self).__init__()
        
        self.pixel_size = _list(pixel_size)
        self.pixel_num = _list(pixel_num)
        
        self.poisson_m = poisson_m
        self.gaussian_std = gaussian_std
        
        lamb0 = torch.tensor(lamb0)
        lamb = lamb0/refractive_index
        self.refractive_index = refractive_index
        
        self.Lx = pixel_size[0] * pixel_num[0]
        self.Ly = pixel_size[1] * pixel_num[1]
        
        x_grid, y_grid = self.set_grid()
        fx_grid, fy_grid = self.set_fgrid()
        
        self.register_buffer('lamb0', lamb0)
        self.register_buffer('lamb', lamb)
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('fx_grid', fx_grid)
        self.register_buffer('fy_grid', fy_grid)
        
        
    
    def init_grid_params(self):
        
        self.Lx = self.pixel_size[0] * self.pixel_num[0]
        self.Ly = self.pixel_size[1] * self.pixel_num[1]
        
        self.x_grid, self.y_grid = self.set_grid()
        self.fx_grid, self.fy_grid = self.set_fgrid()
    
    def set_grid(self):
        ## center : 0
        return set_spatial_grid(self.pixel_num[0], self.pixel_num[1], self.pixel_size[0], self.pixel_size[1])
    
    def set_fgrid(self):
        # define those with frequency instead of wavevector
        return set_freq_grid(self.pixel_num[0], self.pixel_num[1], self.pixel_size[0], self.pixel_size[1])

    # def sas_rev_calculate_grids(self, zs: list):
    #     ### grid and f_grid are already defined as the sensor grid:
    #     """
    #     Args:
    #         zs (list): The propagation distance used in [SASPropagation].
    #                     (from starting point to end point in optical system.)
    #     """
    #     zs = _list(zs)
        
    #     def calculate_sas_prev_grid(H, W, dx, dy, z):
    #         """
    #         Args: The grid parameters of scalable ASM's output.
    #             H (int): Pixel num
    #             W (int): Pixel num
    #             dx (float): Pixel size
    #             dy (float): Pixel size
    #             z (float): Propagation distance
    #         """
    #         prev_Lx = self.lamb * z * H / (2 * H * dx)
    #         prev_Ly = self.lamb * z * W / (2 * W * dy)
    #         prev_dx, prev_dy = prev_Lx/H, prev_Ly/W
    #         return prev_Lx, prev_Ly, prev_dx, prev_dy
            
    #     ##### sensor layer's grid (Target grid)
    #     Lx, Ly = self.Lx, self.Ly
    #     H, W = self.pixel_num[0], self.pixel_num[1]
    #     dx, dy = self.pixel_size[0], self.pixel_size[1]
    #     for z in reversed(zs): # last부터 거꾸로...
    #         Lx, Ly, dx, dy = calculate_sas_prev_grid(H, W, dx, dy, z)
        
    #     ### make source grid
    #     self.pixel_size = [dx, dy]
    #     self.Lx, self.Ly = Lx, Ly
        
    #     self.x_grid, self.y_grid = self.set_grid()
    #     self.fx_grid, self.fy_grid = self.set_fgrid()
        #### Setting self.grid and self.f_grid as Source's grid and f_grid
        #### After applying several scalable ASM, then the functions would outcome each output's grid.
        


"""
    This instance is inspired by https://github.com/chromatix-team/chromatix/blob/main/src/chromatix/field.py
    In order to allocate the torch.tensor data to gpu devices, it is better to use these tensors which will be
    allocated to devices in forward process instead of the initialization process.
    Comprehensively, there are four tensors that are allocated to gpu devices.
    1. field
    2. wavelengths
    3. spatial domain grid (x_grid, y_grid)
    4. frequency domain grid (fx_grid, fy_grid)
"""

class Field:
    def __init__(self,
        field=None,
        lamb0=None,
        x_grid=None,
        y_grid=None,
        fx_grid=None,
        fy_grid=None,
        field_init_path=None ### only consider npy file
    ):
        """
        Args:
            field (torch.Tensor): (1, C, H, W).
            lamb0 (torch.Tensor): Wavelength in free space (C)
            x_grid, y_grid (torch.Tensor): Spatial domain grid (H, 1), (1, W).
            fx_grid, fy_grid (torch.Tensor): Frequency domain grid (H, 1), (1, W)..
            device (str or torch.device): 
        """
        for name, tensor in [("field", field), ("lamb0", lamb0), ("x_grid", x_grid), 
                             ("y_grid", y_grid), ("fx_grid", fx_grid), ("fy_grid", fy_grid)]:
            if tensor is not None and not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        
        self.field = field
        self.lamb0 = lamb0
        
        """
        We set the grid parameters as the low-rank form to mitigate the use of larger memory.
        Sometimes, in-place operation is not desirable since it can influence computational graphs.
        Keeping this in mind, I try to minimize the allocation of memory caused by non-in-place operations
        by setting the grid parameters as the low-rank form.
        """
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.fx_grid = fx_grid
        self.fy_grid = fy_grid
        
        self.phase_applied = False 
        field_init_flag = False if field_init_path == None else True
        self.field_init_path = field_init_path
        if field_init_flag:
            self.field_init_with_path()
    
    def field_init_with_path(self):
        loaded_field = torch.from_numpy(np.load(self.field_init_path))
        H, _ = self.x_grid.shape
        _, W = self.y_grid.shape
        assert loaded_field.shape[-2] == H and loaded_field.shape[-2] == W, "The shape of loaded field should be same with that of the defined grid."
        self.field = loaded_field
        

    def to(self, device):
        device = torch.device(device)
        self.field = self.field.to(device) if self.field is not None else None
        self.lamb0 = self.lamb0.to(device) if self.lamb0 is not None else None
        self.x_grid = self.x_grid.to(device) if self.x_grid is not None else None
        self.y_grid = self.y_grid.to(device) if self.y_grid is not None else None
        self.fx_grid = self.fx_grid.to(device) if self.fx_grid is not None else None
        self.fy_grid = self.fy_grid.to(device) if self.fy_grid is not None else None
        return self

    def replace_field(self, field):
        return Field(
            field=field,
            lamb0=self.lamb0,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            fx_grid=self.fx_grid,
            fy_grid=self.fy_grid,
        )


    def unsqueeze(self, dim):
        """New field instance"""
        return self.replace_field(self.field.unsqueeze(dim))

    def __abs__(self):
        """새로운 Field 인스턴스를 생성 (non-in-place)."""
        return self.replace_field(torch.abs(self.field)) 
    
    def abs(self):
        return self.__abs__()

    def __add__(self, b):
        """새로운 Field 인스턴스를 생성하여 반환 (non-in-place)."""
        if self.field is None:
            raise ValueError("self.field is None, cannot perform addition")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform addition")
            new_field = self.field + b.field
        else:
            new_field = self.field + b
        return self.replace_field(new_field)
    
    def __radd__(self, b):
        if self.field is None:
            raise ValueError("self.field is None, cannot perform addition")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform addition")
            new_field = b.field.to(self.device) + self.field
        else:
            new_field = b + self.field
        return self.replace_field(new_field)

    def __iadd__(self, b):
        """In-place Addition Warning when tracking gradient"""
        if self.field is None:
            raise ValueError("self.field is None, cannot perform in-place addition")
        if self.field.requires_grad:
            warnings.warn("In-place operation on a tensor with requires_grad=True may break the computational graph. Consider using non-in-place operation (+).")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform in-place addition")
            self.field = self.field + b.field
        else:
            self.field = self.field + b
        return self

    def __mul__(self, b):
        if self.field is None:
            raise ValueError("self.field is None, cannot perform multiplication")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform multiplication")
            new_field = self.field * b.field
        else:
            new_field = self.field * b
        return self.replace_field(new_field)
    
    def __rmul__(self, b):
        if self.field is None:
            raise ValueError("self.field is None, cannot perform multiplication")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform multiplication")
            new_field = b.field.to(self.device) * self.field
        else:
            new_field = b * self.field
        return self.replace_field(new_field)

    def __imul__(self, b):
        """In-place Multiplication. Warning when tracking gradient"""
        if self.field is None:
            raise ValueError("self.field is None, cannot perform in-place multiplication")
        if self.field.requires_grad:
            warnings.warn("In-place operation on a tensor with requires_grad=True may break the computational graph. Consider using non-in-place operation (*).")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform in-place multiplication")
            self.field = self.field * b.field
        else:
            self.field = self.field * b
        return self

    def __truediv__(self, b):
        if self.field is None:
            raise ValueError("self.field is None, cannot perform division")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform division")
            new_field = self.field / b.field
        else:
            new_field = self.field / b
        return self.replace_field(new_field)
    
    def __rtruediv__(self, b):
        if self.field is None:
            raise ValueError("self.field is None, cannot perform division")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform division")
            new_field = b.field.to(self.device) / self.field
        else:
            new_field = b / self.field
        return self.replace_field(new_field)


    def __itruediv__(self, b):
        """In-place Divide. Warning when tracking gradient"""
        if self.field is None:
            raise ValueError("self.field is None, cannot perform in-place division")
        if self.field.requires_grad:
            warnings.warn("In-place operation on a tensor with requires_grad=True may break the computational graph. Consider using non-in-place operation (/).")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform in-place division")
            self.field = self.field / b.field
        else:
            self.field = self.field / b
        return self

    def __sub__(self, b):
        if self.field is None:
            raise ValueError("self.field is None, cannot perform subtraction")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform subtraction")
            new_field = self.field - b.field
        else:
            new_field = self.field - b
        return self.replace_field(new_field)
    
    def __rsub__(self, b):
        if self.field is None:
            raise ValueError("self.field is None, cannot perform subtraction")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform subtraction")
            new_field = b.field.to(self.device) - self.field
        else:
            new_field = b - self.field
        return self.replace_field(new_field)

    def __isub__(self, b):
        if self.field is None:
            raise ValueError("self.field is None, cannot perform in-place subtraction")
        if self.field.requires_grad:
            warnings.warn("In-place operation on a tensor with requires_grad=True may break the computational graph. Consider using non-in-place operation (-).")
        if isinstance(b, Field):
            if b.field is None:
                raise ValueError("b.field is None, cannot perform in-place subtraction")
            self.field = self.field - b.field
        else:
            self.field = self.field - b
        return self

    @property
    def device(self):
        return self.field.device if self.field is not None else None

    @property
    def shape(self):
        return self.field.shape if self.field is not None else None

    @property
    def dx(self):
        return abs(self.x_grid[1, 0] - self.x_grid[0, 0]) if self.x_grid is not None else None

    @property
    def dy(self):
        return abs(self.y_grid[0, 1] - self.y_grid[0, 0]) if self.y_grid is not None else None
    
    @property
    def Lx(self):
        return self.dx * self.field[-2]

    @property
    def Ly(self):
        return self.dy * self.field[-1]

    @property
    def dfx(self):
        return abs(self.fx_grid[1, 0] - self.fx_grid[0, 0]) if self.fx_grid is not None else None

    @property
    def dfy(self):
        return abs(self.fy_grid[0, 1] - self.fy_grid[0, 0]) if self.fy_grid is not None else None

    @property
    def pixel_area(self):
        return self.dx * self.dy if self.dx is not None and self.dy is not None else None

    @property
    def freq_pixel_area(self):
        return self.dfx * self.dfy if self.dfx is not None and self.dfy is not None else None

    @property
    def intensity(self):
        return compute_intensity(self.field, sum=True) if self.field is not None else None

    @property
    def ch_intensity(self):
        return compute_intensity(self.field, sum=False) if self.field is not None else None

    @property
    def power(self):
        return compute_power(self.field, self.pixel_area) if self.field is not None and self.pixel_area is not None else None
    