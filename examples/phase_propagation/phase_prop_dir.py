# Considering direction of the field
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
from systems.utils import _pair, NA2D
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy.special import fresnel

def nyquist_pixelsize_criterion(NA, lamb):
    max_pixel_size = lamb/(2*NA) # C, Tensor
    return torch.min(max_pixel_size).item() # bound..


def x_angle_to_dir_factor(degree):
    theta = math.pi/180 * degree
    return [math.sin(theta), 0, math.cos(theta)]

def xy_angle_to_dir_factor(degree):
    theta = math.pi/180 * degree
    return [math.sin(theta), math.sin(theta), math.cos(theta)]

class LensPropagation(OpticalSystem):
    def __init__(
        self,
        pixel_size=[1, 1],
        pixel_num=[1000, 1000],
        lamb0=[0.55, 1.05, 1.550],
        refractive_index=1,
        z=19*1e3,
        NA=0.3,
        dir_factors=None,
        nyquist_spatial_bound=True,
        degree=None,
    ):
        super(LensPropagation, self).__init__(
            pixel_size=pixel_size,
            pixel_num=pixel_num,
            lamb0=lamb0,
            refractive_index=refractive_index
        )
        self.z = z
        D = NA2D(NA, focal_length=z, ref_idx=refractive_index)
        
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
            dir_factors=dir_factors,
            power=1.0,
        )
        # self.pupil_mask = elem.CirclePupil(self.x_grid, self.y_grid, pupil_width) if pupil_type=='circle' else elem.SquarePupil(self.x_grid, self.y_grid, pupil_width)
        self.mid_lens = elem.MultConvergingLens(ref_idx=self.refractive_index, focal_length=z, D=D)
        
        
        self.prop = elem.ASMPropagation(
            z=z,
            ref_idx=self.refractive_index,
            band_limited=True
        )
        
        self.sensor = elem.Sensor(shot_noise_modes=[], clip=[1e-20, 1e+9], channel_sum=False)    
    
    def forward(self):
        
        field = Field(lamb0=self.lamb0, x_grid=self.x_grid, y_grid=self.y_grid, fx_grid=self.fx_grid, fy_grid=self.fy_grid)
        
        src_field = self.source(field)
        print(f"Initial Field's shape: {src_field.shape}")
        pupiled_field = self.mid_lens(src_field)
        print(f"mid_lens: {pupiled_field.shape}")    
        prop_field = self.prop(pupiled_field) # asm
        if isinstance(prop_field, list) or isinstance(prop_field, tuple): # SASPropagation.
            prop_field, pixel_size = prop_field
        print(f"Field's shape after propagation: {prop_field.shape}")
        out = self.sensor(prop_field)
        print(f"Output Field's shape : {out.shape}")
        return src_field.field, pupiled_field.field, prop_field.field, out
    

def make_kwargs(dir_factors, degree, lamb0, z=None):
    this_kwargs = dict(
        pixel_size=[0.5, 0.5],
        pixel_num=[500, 500],
        lamb0=[lamb0],
        refractive_index=1,
        z=1*1e3 if z == None else z,
        NA=0.5,
        dir_factors=dir_factors,
        nyquist_spatial_bound=True,
        degree=degree
    )
    return this_kwargs
    
def iterative_perform_(kwargss: list, device):
    outs = {}
    sqr_fresnel = {}
    for kwargs in kwargss:
        Prop = LensPropagation(**kwargs).to(device)
        src_field, pupiled_field, prop_field, out = Prop()
        #### Visualization code.
        #### Out field instantly save in list.
        #### Show the figures in grid!
        out = out.detach().cpu()
        lamb0 = kwargs['lamb0']
        z = Prop.z
        dir_factors = kwargs['dir_factors']
        degree = kwargs['degree']
            
        dict_key = f"deg{degree}_lamb0{lamb0[0]}"
        outs[dict_key] = out
    
    return outs

def iterative_perform(save_root, file_name, device):
    lamb0s = [0.4, 0.55, 0.7]
    degrees = [-6, -3, 3, 6]

    kwargss = []
    for lamb0 in lamb0s:
        for i, degree in enumerate(degrees):
            kwargss.append(make_kwargs(xy_angle_to_dir_factor(degree), degree, lamb0))
    out_dict = iterative_perform_(kwargss, device)
    ### Visualization with varying out_dict[f"NA{NA}_lamb0{lamb0}"]
    ### Make Fig grid
    fig, axes = plt.subplots(nrows=len(lamb0s), ncols=len(degrees))
    round2 = partial(round, 2)
    for i, lamb0 in enumerate(lamb0s):
        for j, degree in enumerate(degrees):
            dict_key = f"deg{degree}_lamb0{lamb0}"
            out = out_dict[dict_key]
            axes[i,j].imshow(torch.abs(out)[0,0])
            axes[i,j].title.set_text("_".join(dict_key.split("_")[:2])) 
            
    plt.tight_layout()
    fig.savefig(os.path.join(save_root, file_name))
    plt.clf()
    
if __name__ == "__main__":
    device = 'cuda'
    
    base_save_root = "./phase_prop_vis/phase_prop_dir"
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
    save_root = os.path.join(base_save_root)
    os.makedirs(save_root, exist_ok=True)
    print(x_angle_to_dir_factor(1))
    Prop = LensPropagation(
            pixel_size=[0.5, 0.5],
            pixel_num=[500, 500],
            lamb0=[0.4, 0.55, 0.7],
            refractive_index=1,
            z=1*1e3,
            NA=0.3,
            dir_factors=x_angle_to_dir_factor(5),# 6deg tilting for x axis
            nyquist_spatial_bound=True
    ).to(device)
    strt = time.time()
    src_field, pupiled_field, prop_field, out = Prop()   
    end = time.time()
    
    print(f"{device} device : {end-strt} (s)")
    x_grid, y_grid = Prop.x_grid, Prop.y_grid
    radial_grid = x_grid ** 2 + y_grid ** 2
    lamb0 = Prop.lamb0 # list
        
    
    
    file_name_format = "{}_field_{}.png"
    title_format = "{} field of Lambda : {}"
    

    visualize(file_name_format.format('pupiled_field', torch.round(lamb0[0]/Prop.nanometers)), pupiled_field[0, 0], title=title_format.format("pupiled_field", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('prop_field', torch.round(lamb0[0]/Prop.nanometers)), prop_field[0, 0], title=title_format.format("prop_field", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    
    visualize(file_name_format.format('pupiled_field', torch.round(lamb0[1]/Prop.nanometers)), pupiled_field[0, 1], title=title_format.format("pupiled_field", torch.round(lamb0[1]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('prop_field', torch.round(lamb0[1]/Prop.nanometers)), prop_field[0, 1], title=title_format.format("prop_field", torch.round(lamb0[1]/Prop.nanometers)), mode='abs')
    
    visualize(file_name_format.format('pupiled_field', torch.round(lamb0[2]/Prop.nanometers)), pupiled_field[0, 2], title=title_format.format("pupiled_field", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('prop_field', torch.round(lamb0[2]/Prop.nanometers)), prop_field[0, 2], title=title_format.format("prop_field", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
    
    
    visualize(file_name_format.format('source', torch.round(lamb0[0]/Prop.nanometers)), src_field[0, 0], title=title_format.format("Source", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('source', torch.round(lamb0[1]/Prop.nanometers)), src_field[0, 1], title=title_format.format("Source", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('source', torch.round(lamb0[2]/Prop.nanometers)), src_field[0, 2], title=title_format.format("Source", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')
    
    visualize(file_name_format.format('sensor', torch.round(lamb0[0]/Prop.nanometers)), out[0, 0], title=title_format.format("Sensor", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('sensor', torch.round(lamb0[1]/Prop.nanometers)), out[0, 1], title=title_format.format("Sensor", torch.round(lamb0[1]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('sensor', torch.round(lamb0[2]/Prop.nanometers)), out[0, 2], title=title_format.format("Sensor", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
        
    iterative_perform(base_save_root, 'wvl_dirfactors_grid_fig.png', device=device) 