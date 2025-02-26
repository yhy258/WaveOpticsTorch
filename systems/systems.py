# TODO : 검증 코드 생각.
# TODO : 각도
# TODO : coherent vs incoherent
# TODO : ...


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

from systems.functions.lenses import pointsource, planesource
from systems.functions.propagations import asm_propagation

############################## INTEGRATED OPTICAL SYSTEM.
# only consider scalar field (disturbance, amplitude, ....)
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
        return x * self.pixel_size, y * self.pixel_size
    
    def set_fgrid(self):
        # d : inverse of sampling rate. sampling rate : 1/Lx
        # define those with frequency instead of wavevector
        fx = np.fft.fftshift(np.fft.fftfreq(self.sensor_height_pixels, self.pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(self.sensor_width_pixels, self.pixel_size))
        fx, fy = np.meshgrid(fx, fy)

        f_grid = (torch.from_numpy(fx), torch.from_numpy(fy))
        return f_grid

        
    def init_source(
        self,
        mode: str = 'point',
        z : float = 1.0, # (meter)
        amplitude: complex = 1.,
        power: float = 1.,
        source_kwargs: dict = {}
    ):
        # mode : point, plane, phase (custom)
        # point source - src_loc : list 
        # plane source - angles (theta, phi) ; list

        # source component's output : the phase profile in z distance.
        assert mode in ["point", "plane", "phase"], f"The parameter mode should be in {['point', 'plane', 'phase']}"
        
        # source_func input : amplitude, z, power
        if mode.lower() == "point":
            field = pointsource(
                grid=self.grid,
                amplitude=amplitude,
                lamb0=self.lamb0,
                z=z,
                ref_idx=self.refractive_index,
                src_loc=source_kwargs["src_loc"],
                power=power,
            )
            # source_func = partial(pointsource, grid=self.grid, lamb0=self.lamb0, ref_idx=self.refractive_index, src_loc=source_kwargs["src_loc"])
        elif mode.lower() == "plane":
            # planewave - option : angle of ... (kphase.)
            field = planesource(
                grid=self.grid,
                amplitude=amplitude,
                lamb0=self.lamb0,
                ref_idx=self.refractive_index,
                dir_factors=source_kwargs["dir_factors"],
                power=power
            )
            # source_func = partial(planesource, grid=self.grid, lamb0=self.lamb0, ref_idx=self.refractive_index, dir_factors=source_kwargs["dir_factors"], dir_mode=source_kwargs["dir_mode"])
        # elif mode.lower() == "phase":
        #     # 임의의 phase가 들어왔을때, 단순 propagation ㅇㅇ..
        #     pass
        else:
            raise Exception(f"Source type setting error. MODE: {mode}")
        
        return field
    
    def propagation(
        self,
        input_field: Tensor, # B, C, H, W or B, C, D, H, W
        z,
        prop_type: str,
        band_limited: bool,
    ):
        # ASM, Rayleigh
        if prop_type.lower() == "asm":
            # band-limited ..
            out_field = asm_propagation(
                input_field=input_field,
                z=z,
                n=self.refractive_index,
                lamb0=self.lamb0,
                Lx=self.Lx,
                Ly=self.Ly,
                fx=self.f_grid[0],
                fy=self.f_grid[1],
                band_limited=band_limited
            )
        # elif prop_type.lower() == "rayleigh_sommerfeld":
        #     pass     
        else:
            raise Exception(f"Prop type setting error. PROP: {prop_type}")
        
        return out_field
    
    def imaging(self, obj=None):
        if obj == None:
            ### only output psf (but intensity?)
            ### maybe depending on the coherent or incoherent illumination setting.
            pass
        else:
            ### psf -> sliding window.. (lets use fftnconv)
            pass
    
    ## 중간에 component 넣는식?

