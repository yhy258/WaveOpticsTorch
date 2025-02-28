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

############################## INTEGRATED OPTICAL SYSTEM.
# only consider scalar field (disturbance, amplitude, ....)

"""
    TODO: PILOT SYSTEM ; 4F-system.
        - Wrap the functionals with class!.
    
    TODO: 
    1. Source 입력 O
    2. Component [렌즈, Phase Mask, Pupil, ...] 혹은 Propagation 입력
    3. Sensor 단 관측 O
    
"""


class OpticalSystem(nn.Module):
    # unit : m
    # do not care "magnification"
    micrometers = 1e-6
    nanometers = 1e-9
    
    def __init__(
        self,
        pixel_size: float = 5e-6,
        sensor_height_pixels: int = 1000,
        sensor_width_pixels: int = 1000,
        ### noise parameters
        poisson_m: float = 0.,
        gaussian_std: float = 1.,
        ### wavelength in free space
        lamb0: list = [], # chromaticity, meter.
        refractive_index: float = 1., # Air = 1.
    ):
        super(OpticalSystem, self).__init__()
        self.pixel_size = pixel_size
        self.sensor_height_pixels = sensor_height_pixels
        self.sensor_width_pixels = sensor_width_pixels
        
        self.poisson_m = poisson_m
        self.gaussian_std = gaussian_std
        
        self.lamb0 = torch.tensor(lamb0)
        self.refractive_index = refractive_index
        
        self.Lx = pixel_size * sensor_height_pixels
        self.Ly = pixel_size * sensor_width_pixels
        
        grid = self.set_grid()
        f_grid = self.set_fgrid()
        
        self.register_buffer("lamb0", torch.tensor(lamb0)) # C,
        self.register_buffer("grid", grid)
        self.register_buffer("f_grid", f_grid)
    
    def set_grid(self):
        ## center : 0
        Nx = self.sensor_height_pixels
        Ny = self.sensor_width_pixels
        x, y = np.meshgrid(
            np.linspace(0, (Nx-1), Nx) - Nx / 2,
            np.linspace(0, (Ny-1), Ny) - Ny / 2,
        )
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return torch.stack(x * self.pixel_size, y * self.pixel_size, dim=0) # 2, H, W
    
    def set_fgrid(self):
        # d : inverse of sampling rate. sampling rate : 1/Lx
        # define those with frequency instead of wavevector
        fx = np.fft.fftshift(np.fft.fftfreq(self.sensor_height_pixels, self.pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(self.sensor_width_pixels, self.pixel_size))
        fx, fy = np.meshgrid(fx, fy)

        f_grid = (torch.from_numpy(fx), torch.from_numpy(fy))
        return torch.stack(f_grid, dim=0) # 2, H< W



class Optical4FSystem(OpticalSystem):
    def __init__(
        self,
        pixel_size: float = 5e-6,
        sensor_height_pixels: int = 1000,
        sensor_width_pixels: int = 1000,
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
            sensor_height_pixels=sensor_height_pixels,
            sensor_width_pixels=sensor_width_pixels,
            poisson_m=poisson_m,
            gaussian_std=gaussian_std,
            lamb0=lamb0,
            refractive_index=refractive_index
        )
        
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
        
        ### PHASE MASK
        
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
        
    def forward(self):
        
        field = self.source()
        field = self.lens_multp_phase(field)
        field = self.prop(field)
        ## phase mask
        
        field = self.fflens(field)
        
        ### imaging
        return self.sensor(field)