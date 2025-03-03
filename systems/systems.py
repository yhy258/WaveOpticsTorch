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
        
        

# Do not care this 4F system code. The example of spatially filtered PSF is a 4f-system.
class Optical4FSystem(OpticalSystem):
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
        focal_length: float = None, # (m)
        NA : float = None,
    ):
        super(Optical4FSystem, self).__init__(
            pixel_size=pixel_size,
            pixel_num=pixel_num,
            poisson_m=poisson_m,
            gaussian_std=gaussian_std,
            lamb0=lamb0,
            refractive_index=refractive_index
        )
        
        # Define inversely scaled source grid from target sensor grid
        # self.sas_rev_calculate_grids(zs=[focal_length]) 
        
        self.focal_length = focal_length
        
        #### Define sources or elements.
        self.source = elem.PointSource(
                grid=self.grid,
                amplitude=1.0,
                lamb0=self.lamb0,
                ref_idx=self.refractive_index,
                z=focal_length,
                src_loc=None, # center.
                power=1.0,
                paraxial=True
            )
        
        self.lens_multp_phase = elem.MultConvergingLens(
            grid=self.grid,
            lamb0=self.lamb0,
            ref_idx=self.refractive_index,
            focal_length=focal_length,
            NA=NA
        )
        
        self.prop = elem.ASMPropagation(
            z=focal_length,
            lamb0=self.lamb0,
            ref_idx=self.refractive_index,
            Lx=self.Lx,
            Ly=self.Ly,
            f_grid=self.f_grid,
            band_limited=True
        )
        
        ### If we use SASPropagation, we have to inversely calculate the source grid from target sensor grid.
        # self.prop = elem.SASPropagation(
        #     grid=self.grid,
        #     z=focal_length,
        #     lamb0=self.lamb0,
        #     ref_idx=self.refractive_index,
        #     Lx=self.Lx,
        #     Ly=self.Ly,
        # )
        
        
        ### PHASE MASK
        # HOW CAN I INITIALIZE THIS ONE? - Zernike option.
        # This should be trainable.
        self.phase_mask = elem.PhaseMaskZernikeInit(grid=self.grid)
        
        ### FF lens
        self.fflens = elem.FFLens(
            grid=self.grid,
            f_grid=self.f_grid,
            lamb0=self.lamb0,
            ref_idx=self.refractive_index,
            focal_length=focal_length,
            NA=NA
        )
        
        ### Sensor
        self.sensor = elem.Sensor(
            shot_noise_modes=["approximated_poisson", "gaussian"]
        )
        
    def get_psf(self):
        # Make point source
        field = self.source()
        ## converging spherical lens's multiplicative phase transformation.
        field = self.lens_multp_phase(field)
        
        ## Free-space propagation
        field = self.prop(field)
        
        ## phase mask
        field = self.phase_mask(field)
        
        # Fourier dom -> Spatial dom
        field = self.fflens(field)
        
        ### imaging - emulating noises.
        return self.sensor(field)
    
    def imaging(self, obj, psf):
        # TODO:
        pass
        
    def forward(self):
        
        pass