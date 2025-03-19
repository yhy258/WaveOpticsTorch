import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)
import math
from functools import partial
import numpy as np
import torch
from torch.functional import Tensor
import systems.elements as elem
from systems.systems import OpticalSystem, Field
from systems.utils import _pair
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy.special import fresnel


def nyquist_pixelsize_criterion(NA, lamb):
    max_pixel_size = lamb/(2*NA) # C, Tensor
    return torch.min(max_pixel_size).item() # bound..


class SinAmplitudeImage(OpticalSystem):
    def __init__(
        self,
        pixel_size=[1, 1],
        pixel_num=[1000, 1000],
        lamb0=[0.55, 1.05, 1.550],
        grating_frequency_factor=None,
        refractive_index=1,
        propagation_distance=19*1e3,
        NA=0.3,
        nyquist_spatial_bound=True,
    ):
        super(SinAmplitudeImage, self).__init__(
            pixel_size=pixel_size,
            pixel_num=pixel_num,
            lamb0=lamb0,
            refractive_index=refractive_index
        )
        self.propagation_distance = propagation_distance
        
        max_pixel_size = nyquist_pixelsize_criterion(NA, self.lamb0/self.refractive_index)
        print("Max Pixel Size : ", max_pixel_size)
        print("Now Pixel Size : ", pixel_size)
        if len(lamb0) == 1:
            print("Now Wavelength : ", lamb0[0])
        if max(pixel_size) > max_pixel_size and nyquist_spatial_bound:
            pixel_size = _pair(max_pixel_size)
            self.pixel_size = pixel_size
            self.init_grid_params()
        
        self.source = elem.PlaneSource(
            amplitude=1.0,
            ref_idx=self.refractive_index,
            dir_factors=None, # center.
            power=1.0,
        )
        
        # Vertical oscillation
        
        m = 1
        period = self.Ly / grating_frequency_factor 
        pupil = 1/2 * (1 + m*torch.cos(2*math.pi*self.y_grid/period)) # 1, W tensor
        self.pupil_mask = elem.CustomPupil(pupil)
        # self.pupil_mask = elem.CirclePupil(self.x_grid, self.y_grid, pupil_width) if pupil_type=='circle' else elem.SquarePupil(self.x_grid, self.y_grid, pupil_width)
        
        self.prop = elem.ASMPropagation(
            z=propagation_distance,
            ref_idx=self.refractive_index,
            band_limited=True
        )
        
        self.sensor = elem.Sensor(shot_noise_modes=[], clip=[1e-20, 1e+9], channel_sum=False)    
    
    def forward(self):
        
        field = Field(lamb0=self.lamb0, x_grid=self.x_grid, y_grid=self.y_grid, fx_grid=self.fx_grid, fy_grid=self.fy_grid)
        
        src_field = self.source(field)
        print(f"Initial Field's shape: {src_field.shape}")
        pupiled_field = self.pupil_mask(src_field)
        print(f"pupil: {pupiled_field.shape}")    
        prop_field = self.prop(pupiled_field) # asm
        if isinstance(prop_field, list) or isinstance(prop_field, tuple): # SASPropagation.
            prop_field, pixel_size = prop_field
        print(f"Field's shape after propagation: {prop_field.shape}")
        out = self.sensor(prop_field)
        print(f"Output Field's shape : {out.shape}")
        return src_field.field, pupiled_field.field, prop_field.field, out
    

"""
    1. Talbot images s.t. z = (2nL^2) / lamb
    2. Phase reversed version of Talbot images s.t. z = (2n+1)L^2 / lamb
    3. Sub-talbot images s.t. z = (n-1/2)L^2 / lamb
"""
    
        
if __name__ == "__main__":
    device = 'cuda'
    
    base_save_root = "./phase_prop_vis/talbot"
    os.makedirs(base_save_root, exist_ok=True)
    
    ### visualize function
    def visualize(file_name, field, title, mode='phase'):
        if field.device != 'cpu':
            field = field.detach().cpu()
        if mode == "abs":
            plt.imshow(torch.abs(field))    
        elif mode == "phase" or mode == 'angle':
            plt.imshow(torch.angle(field))
        elif mode == "real":
            plt.imshow(field.real)
        plt.title(title + "_" + mode)
        plt.colorbar()
        plt.savefig(os.path.join(save_root, file_name))
        plt.clf()
    grating_frequency_factor = 10
    
    sample_prop = SinAmplitudeImage(
            pixel_size=[0.5, 0.5],
            pixel_num=[500, 500],
            lamb0=[0.55],
            grating_frequency_factor=grating_frequency_factor,
            refractive_index=1,
            propagation_distance=10*1e3,
            NA=0.3,
            nyquist_spatial_bound=False
    )
    period = sample_prop.Ly / grating_frequency_factor 
    talbot_name_dict = {
        "talbot": (period) ** 2 * 2*math.pi / (math.pi * sample_prop.lamb0[0]),
        "revtalbot": (period) ** 2 * 3 * math.pi / (math.pi * sample_prop.lamb0[0]),
        "subtalbot": (period) ** 2 * 1/2 * math.pi / (math.pi * sample_prop.lamb0[0]),
        }
    for k, v in talbot_name_dict.items():    
        save_root = os.path.join(base_save_root)
        os.makedirs(save_root, exist_ok=True)
        Prop = SinAmplitudeImage(
                pixel_size=[0.5, 0.5],
                pixel_num=[500, 500],
                lamb0=[0.55],
                grating_frequency_factor=grating_frequency_factor,
                refractive_index=1,
                propagation_distance=v,
                NA=0.3,
                nyquist_spatial_bound=False
        ).to(device)
        
        strt = time.time()
        src_field, pupiled_field, prop_field, out = Prop()   
        end = time.time()
        
        print(f"{device} device : {end-strt} (s)")
        x_grid, y_grid = Prop.x_grid, Prop.y_grid
        radial_grid = x_grid ** 2 + y_grid ** 2
        lamb0 = Prop.lamb0 # list
        
        file_name_format = "{}_{}_field_{}.png"
        title_format = "{} {} field of Lambda : {}, z (µm): {}"
        

        visualize(file_name_format.format(k, 'pupiled_field', torch.round(lamb0[0]/Prop.nanometers)), pupiled_field[0, 0], title=title_format.format(k, "pupiled_field", torch.round(lamb0[0]/Prop.nanometers), torch.round(Prop.propagation_distance)), mode='abs')
        visualize(file_name_format.format(k, 'prop_field', torch.round(lamb0[0]/Prop.nanometers)), prop_field[0, 0], title=title_format.format(k, "prop_field", torch.round(lamb0[0]/Prop.nanometers), torch.round(Prop.propagation_distance)), mode='abs')
        
        visualize(file_name_format.format(k, 'source', torch.round(lamb0[0]/Prop.nanometers)), src_field[0, 0], title=title_format.format(k, "Source", torch.round(lamb0[0]/Prop.nanometers), torch.round(Prop.propagation_distance)), mode='angle')
        
        visualize(file_name_format.format(k, 'sensor', torch.round(lamb0[0]/Prop.nanometers)), out[0, 0], title=title_format.format(k, "Sensor", torch.round(lamb0[0]/Prop.nanometers), torch.round(Prop.propagation_distance)), mode='abs')
        
        
    visualize(file_name_format.format('pupilmask', 'sensor', torch.round(lamb0[0]/Prop.nanometers)), Prop.pupil_mask.mask.detach().cpu().repeat(repeats=(500,1)), title=title_format.format(k, "Pupilmask", torch.round(lamb0[0]/Prop.nanometers), Prop.propagation_distance), mode='abs')
