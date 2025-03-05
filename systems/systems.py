"""
    COHERENT vs INCOHERENT
    1. COHERENT ILLUMINATION : ATF (the relationship between output amplitude and input ampliutde is linear with amplitude transfer function.)
        F{AMP_OUTPUT} : ATF*F{AMP_INPUT}
        |AMP_OUTPUT|^2 is detecteed on sensor.
    2. INCOHERENT ILLUMINATION : OTF (the relationship between input intensity and output intensity is linear with optical transfer function; auto-correlation of ATF)
        F{INT_OUTPUT} = OTF * F{INT_INPUT}
        INT_OUTPUT is detected on sensor.
"""




import os, sys, math, datetime, glob, faulthandler
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor

from functools import partial

import systems.elements as elem
from utils import set_spatial_grid, set_freq_grid, _list

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
        
        self.lamb0 = torch.tensor(lamb0)
        self.refractive_index = refractive_index
        self.lamb = self.lamb0/self.refractive_index
        
        self.Lx = pixel_size[0] * pixel_num[0]
        self.Ly = pixel_size[1] * pixel_num[1]
        
        self.grid = self.set_grid()
        self.f_grid = self.set_fgrid()
    
    def init_grid_params(self):
        
        self.Lx = self.pixel_size[0] * self.pixel_num[0]
        self.Ly = self.pixel_size[1] * self.pixel_num[1]
        
        self.grid = self.set_grid()
        self.f_grid = self.set_fgrid()
    
    def set_grid(self):
        ## center : 0
        return set_spatial_grid(self.pixel_num[0], self.pixel_num[1], self.pixel_size[0], self.pixel_size[1])
    
    def set_fgrid(self):
        # define those with frequency instead of wavevector
        return set_freq_grid(self.pixel_num[0], self.pixel_num[1], self.pixel_size[0], self.pixel_size[1])

    def sas_rev_calculate_grids(self, zs: list):
        ### grid and f_grid are already defined as the sensor grid:
        """
        Args:
            zs (list): The propagation distance used in [SASPropagation].
                        (from starting point to end point in optical system.)
        """
        zs = _list(zs)
        
        def calculate_sas_prev_grid(H, W, dx, dy, z):
            """
            Args: The grid parameters of scalable ASM's output.
                H (int): Pixel num
                W (int): Pixel num
                dx (float): Pixel size
                dy (float): Pixel size
                z (float): Propagation distance
            """
            prev_Lx = self.lamb * z * H / (2 * H * dx)
            prev_Ly = self.lamb * z * W / (2 * W * dy)
            prev_dx, prev_dy = prev_Lx/H, prev_Ly/W
            return prev_Lx, prev_Ly, prev_dx, prev_dy
            
        ##### sensor layer's grid (Target grid)
        Lx, Ly = self.Lx, self.Ly
        H, W = self.pixel_num[0], self.pixel_num[1]
        dx, dy = self.pixel_size[0], self.pixel_size[1]
        for z in reversed(zs): # last부터 거꾸로...
            Lx, Ly, dx, dy = calculate_sas_prev_grid(H, W, dx, dy, z)
        
        ### make source grid
        self.pixel_size = [dx, dy]
        self.Lx, self.Ly = Lx, Ly
        
        self.grid = self.set_grid()
        self.f_grid = self.set_fgrid()
        #### Setting self.grid and self.f_grid as Source's grid and f_grid
        #### After applying several scalable ASM, then the functions would outcome each output's grid.
        
        