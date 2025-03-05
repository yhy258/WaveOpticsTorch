# Considering direction of the field

import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)
from functools import partial
import numpy as np
import torch
import systems.elements as elem
from systems.systems import OpticalSystem
from systems.utils import _pair
import matplotlib.pyplot as plt


def nyquist_pixelsize_criterion(NA, lamb):
    max_pixel_size = lamb/(2*NA) # C, Tensor
    return torch.min(max_pixel_size).item() # bound..
    

class Diffraction(OpticalSystem):
    def __init__(
        self,
        pixel_size=[1, 1],
        pixel_num=[1000, 1000],
        lamb0=[0.55, 1.05, 1.550],
        refractive_index=1,
        paraxial=False,
        focal_length=19*1e3,
        NA=0.3,
        pupil_type='circle',
        pupil_width=20,
        nyquist_spatial_bound=True,
    ):
        super(Diffraction, self).__init__(
            pixel_size=pixel_size,
            pixel_num=pixel_num,
            lamb0=lamb0,
            refractive_index=refractive_index
        )
        self.pupil_type = pupil_type
        
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
            grid=self.grid,
            amplitude=1.0,
            lamb0=self.lamb0,
            ref_idx=self.refractive_index,
            dir_factors=None, # center.
            power=1.0,
        )
        
        
        self.pupil_mask = elem.circular_pupil(self.grid, pupil_width) if pupil_type == 'circle' else elem.square_pupil(self.grid, pupil_width,)
        
        self.prop = elem.ASMPropagation(
            z=focal_length,
            lamb0=self.lamb0,
            ref_idx=self.refractive_index,
            dx=self.pixel_size[0],
            dy=self.pixel_size[1],
            band_limited=True
        )
        
        self.sensor = elem.Sensor(shot_noise_modes=[], clip=[1e-20, 1e+9], channel_sum=False)    
    
    def forward(self):
        unpowered_src_field, src_field = self.source()
        H, W = src_field.shape[-2:]
        print(f"Initial Field's shape: {src_field.shape}")
        pupiled_field = self.pupil_mask * src_field
        print(f"Field's shape after {self.pupil_type} pupil: {pupiled_field.shape}")    
        prop_field = self.prop(pupiled_field) # asm
        if isinstance(prop_field, list) or isinstance(prop_field, tuple): # SASPropagation.
            prop_field, pixel_size = prop_field
        print(f"Field's shape after propagation: {prop_field.shape}")
        out = self.sensor(prop_field)
        print(f"Output Field's shape : {out.shape}")
        return src_field, pupiled_field, prop_field, out
    
    
# 다양한 Propagation distance, Wavelengths, Pixelsize, NA에 대해 ㄱㄱ?
# Lambda : 0.4, 0.55, 0.7
# Propagation distance : z
# NA : 0.1, 0.3, 0.5
# Pixel size : Nyquist spatial bound x

def make_kwargs(lamb0, diameter, pupil_type):
    this_kwargs = dict(
        pixel_size=[0.6, 0.6],
        pixel_num=[500, 500],
        lamb0=[lamb0],
        refractive_index=1,
        paraxial=False,
        focal_length=10*1e3,
        NA=0.3,
        pupil_type=pupil_type,
        pupil_width=diameter,
        nyquist_spatial_bound=False
    )
    return this_kwargs
    
def iterative_perform_(kwargss: list):
    outs = {}
    for kwargs in kwargss:
        Prop = Diffraction(**kwargs)
        src_field, pupiled_field, prop_field, out = Prop()
        #### Visualization code.
        #### Out field instantly save in list.
        #### Show the figures in grid!
        diameter = kwargs['pupil_width']
        lamb0 = kwargs['lamb0']
        pt = kwargs['pupil_type']
        dict_key = f"d{diameter}_lamb0{lamb0[0]}_{pt}pup"
        outs[dict_key] = out
    return outs

def iterative_perform(save_root, file_name):
    lamb0s = [0.4, 0.55, 0.7]
    diameters = [50, 100, 200]
    kwargss = []
    pupil_types = ['circle', 'square']
    for pt in pupil_types:
        for lamb0 in lamb0s:
            for diameter in diameters:
                kwargss.append(make_kwargs(lamb0, diameter, pt))
        out_dict = iterative_perform_(kwargss)
        ### Visualization with varying out_dict[f"NA{NA}_lamb0{lamb0}"]
        ### Make Fig grid
        fig, axes = plt.subplots(nrows=len(lamb0s), ncols=len(diameters))
        
        for i, lamb0 in enumerate(lamb0s):
            for j, diameter in enumerate(diameters):
                dict_key = f"d{diameter}_lamb0{lamb0}_{pt}pup"
                out = out_dict[dict_key]
                axes[i,j].imshow(torch.abs(out)[0,0])
                axes[i,j].title.set_text("_".join(dict_key.split("_")[:2])) 
                
        plt.tight_layout()
        fig.savefig(os.path.join(save_root, f"{pt}_"+file_name))
        plt.clf()


if __name__ == "__main__":
    
    
    base_save_root = "./phase_prop_vis/diffraction"
    os.makedirs(base_save_root, exist_ok=True)
    
    ### visualize function
    def visualize(file_name, field, title, mode='phase'):
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
    
    for pupil_type in ['square', 'circle']:
        save_root = os.path.join(base_save_root, pupil_type)
        os.makedirs(save_root, exist_ok=True)
        Prop = Diffraction(
                pixel_size=[0.6, 0.6],
                pixel_num=[1000, 1000],
                lamb0=[0.4, 0.55, 0.7],
                refractive_index=1,
                paraxial=False,
                focal_length=10*1e3,
                NA=0.3,
                pupil_type=pupil_type,
                pupil_width=100,
                nyquist_spatial_bound=False
        )
        src_field, pupiled_field, prop_field, out = Prop()   
        
        this_grid = Prop.grid
        lamb0 = Prop.lamb0 # list
            
        
        
        file_name_format = "{}_pupil_{}_field_{}.png"
        title_format = "{} {} field of Lambda : {}"
        

        visualize(file_name_format.format(pupil_type, 'pupiled_field', torch.round(lamb0[0]/Prop.nanometers)), pupiled_field[0, 0], title=title_format.format(pupil_type, "pupiled_field", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
        visualize(file_name_format.format(pupil_type, 'prop_field', torch.round(lamb0[0]/Prop.nanometers)), prop_field[0, 0], title=title_format.format(pupil_type, "prop_field", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
        
        visualize(file_name_format.format(pupil_type, 'pupiled_field', torch.round(lamb0[1]/Prop.nanometers)), pupiled_field[0, 1], title=title_format.format(pupil_type, "pupiled_field", torch.round(lamb0[1]/Prop.nanometers)), mode='abs')
        visualize(file_name_format.format(pupil_type, 'prop_field', torch.round(lamb0[1]/Prop.nanometers)), prop_field[0, 1], title=title_format.format(pupil_type, "prop_field", torch.round(lamb0[1]/Prop.nanometers)), mode='abs')
        
        visualize(file_name_format.format(pupil_type, 'pupiled_field', torch.round(lamb0[2]/Prop.nanometers)), pupiled_field[0, 2], title=title_format.format(pupil_type, "pupiled_field", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
        visualize(file_name_format.format(pupil_type, 'prop_field', torch.round(lamb0[2]/Prop.nanometers)), prop_field[0, 2], title=title_format.format(pupil_type, "prop_field", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
        
        
        visualize(file_name_format.format(pupil_type, 'source', torch.round(lamb0[0]/Prop.nanometers)), src_field[0, 0], title=title_format.format(pupil_type, "Source", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
        visualize(file_name_format.format(pupil_type, 'source', torch.round(lamb0[1]/Prop.nanometers)), src_field[0, 1], title=title_format.format(pupil_type, "Source", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
        visualize(file_name_format.format(pupil_type, 'source', torch.round(lamb0[2]/Prop.nanometers)), src_field[0, 2], title=title_format.format(pupil_type, "Source", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')
        
        visualize(file_name_format.format(pupil_type, 'sensor', torch.round(lamb0[0]/Prop.nanometers)), out[0, 0], title=title_format.format(pupil_type, "Sensor", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
        visualize(file_name_format.format(pupil_type, 'sensor', torch.round(lamb0[1]/Prop.nanometers)), out[0, 1], title=title_format.format(pupil_type, "Sensor", torch.round(lamb0[1]/Prop.nanometers)), mode='abs')
        visualize(file_name_format.format(pupil_type, 'sensor', torch.round(lamb0[2]/Prop.nanometers)), out[0, 2], title=title_format.format(pupil_type, "Sensor", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
        
    iterative_perform(base_save_root, 'wvl_pupwidth_grid_fig.png')