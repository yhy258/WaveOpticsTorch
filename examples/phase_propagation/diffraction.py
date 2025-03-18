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
from systems.utils import _pair
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy.special import fresnel

def nyquist_pixelsize_criterion(NA, lamb):
    max_pixel_size = lamb/(2*NA) # C, Tensor
    return torch.min(max_pixel_size).item() # bound..


def f_number(width: float, lamb: float, z: float):
    return (width/2)**2 / (lamb* z)

def get_width(f_num, lamb, z):
    return math.sqrt(f_num*lamb*z) * 2

def get_z(f_num, width, lamb):
    return (width/2)**2 / (lamb * f_num)

def fresnel_diffraction_square_aperture(width: float, x_grid: Tensor, y_grid: Tensor, lamb: float, z: float):
    h_width = width/2
    if len(x_grid.shape)> 1:
        x_grid = x_grid.squeeze()
        y_grid = y_grid.squeeze()
    coef = (2/(lamb*z))**(1/2)
    alpha1s, alpha2s = -coef*(h_width + x_grid), coef*(h_width - x_grid)
    beta1s, beta2s = -coef*(h_width + y_grid), coef*(h_width - y_grid)
    aC1, aS1 = fresnel(alpha1s.detach().tolist())
    aC2, aS2 = fresnel(alpha2s.detach().tolist())
    bC1, bS1 = fresnel(beta1s.detach().tolist())
    bC2, bS2 = fresnel(beta2s.detach().tolist())
    aC1, aS1, aC2, aS2, bC1, bS1, bC2, bS2 = map(torch.tensor, [aC1, aS1, aC2, aS2, bC1, bS1, bC2, bS2])
    unsq0 = partial(torch.unsqueeze, dim=0)
    unsq1 = partial(torch.unsqueeze, dim=1)
    aC1, aS1, aC2, aS2 = map(unsq1, [aC1, aS1, aC2, aS2])
    bC1, bS1, bC2, bS2 = map(unsq0, [bC1, bS1, bC2, bS2])
    I = 1/4*((aC2-aC1)**2+(aS2-aS1)**2) * ((bC2-bC1)**2+(bS2-bS1)**2)
    return I

class Diffraction(OpticalSystem):
    def __init__(
        self,
        pixel_size=[1, 1],
        pixel_num=[1000, 1000],
        lamb0=[0.55, 1.05, 1.550],
        refractive_index=1,
        z=19*1e3,
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
        self.z = z
        
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
        self.pupil_mask = elem.CirclePupil(pupil_width) if pupil_type=='circle' else elem.SquarePupil(self.x_grid, self.y_grid, pupil_width)
        
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
        pupiled_field = self.pupil_mask(src_field)
        print(f"Field's shape after {self.pupil_type} pupil: {pupiled_field.shape}")    
        prop_field = self.prop(pupiled_field) # asm
        if isinstance(prop_field, list) or isinstance(prop_field, tuple): # SASPropagation.
            prop_field, pixel_size = prop_field
        print(f"Field's shape after propagation: {prop_field.shape}")
        out = self.sensor(prop_field)
        print(f"Output Field's shape : {out.shape}")
        return src_field.field, pupiled_field.field, prop_field.field, out
    
    
# 다양한 Propagation distance, Wavelengths, Pixelsize, NA에 대해 ㄱㄱ?
# Lambda : 0.4, 0.55, 0.7
# Propagation distance : z
# NA : 0.1, 0.3, 0.5
# Pixel size : Nyquist spatial bound x

def make_kwargs(lamb0, diameter, pupil_type, z=None):
    this_kwargs = dict(
        pixel_size=[0.6, 0.6],
        pixel_num=[500, 500],
        lamb0=[lamb0],
        refractive_index=1,
        z=10*1e3 if z == None else z,
        NA=0.3,
        pupil_type=pupil_type,
        pupil_width=diameter,
        nyquist_spatial_bound=False
    )
    return this_kwargs
    
def iterative_perform_(kwargss: list, device, sqr_fresnel_case=False):
    outs = {}
    sqr_fresnel = {}
    for kwargs in kwargss:
        Prop = Diffraction(**kwargs).to(device)
        src_field, pupiled_field, prop_field, out = Prop()
        #### Visualization code.
        #### Out field instantly save in list.
        #### Show the figures in grid!
        out = out.detach().cpu()
        diameter = kwargs['pupil_width']
        lamb0 = kwargs['lamb0']
        z = Prop.z
        pt = kwargs['pupil_type']
            
        f_num = f_number(diameter, lamb0[0], z)
        dict_key = f"fnum{str(round(f_num, 1))}_z{str(round(z, 1))}_d{str(round(diameter, 1))}_lamb0{lamb0[0]}_{pt}pup"
        outs[dict_key] = out
        
        
        if sqr_fresnel_case:
            fresnel_out = fresnel_diffraction_square_aperture(width=diameter, x_grid=Prop.x_grid, y_grid=Prop.y_grid, lamb=lamb0[0], z=z)
            sqr_fresnel[dict_key] = fresnel_out
    if sqr_fresnel_case:
        return outs, sqr_fresnel
    return outs

def iterative_perform(save_root, file_name, device):
    lamb0s = [0.4, 0.55, 0.7]
    diameters = [50, 100, 200]
    
    pupil_types = ['circle', 'square']
    for pt in pupil_types:
        kwargss = []
        for lamb0 in lamb0s:
            for diameter in diameters:
                kwargss.append(make_kwargs(lamb0, diameter, pt))
        out_dict = iterative_perform_(kwargss, device)
        ### Visualization with varying out_dict[f"NA{NA}_lamb0{lamb0}"]
        ### Make Fig grid
        fig, axes = plt.subplots(nrows=len(lamb0s), ncols=len(diameters))
        z = 10*1e3
        for i, lamb0 in enumerate(lamb0s):
            for j, diameter in enumerate(diameters):
                f_num = f_number(width=diameter, lamb=lamb0, z=z)
                dict_key = f"fnum{str(round(f_num, 1))}_z{str(round(z, 1))}_d{str(round(diameter, 1))}_lamb0{lamb0}_{pt}pup"
                out = out_dict[dict_key]
                axes[i,j].imshow(torch.abs(out)[0,0])
                axes[i,j].title.set_text("_".join(dict_key.split("_")[:2])) 
                
        plt.tight_layout()
        fig.savefig(os.path.join(save_root, f"{pt}_"+file_name))
        plt.clf()
        
### GIF Visualization function    
def animate(i, dict_data, keys, mode='abs'):
    this_key = keys[i]
    out = dict_data[this_key]
    plt.cla()
    
    if mode == 'abs':
        plt.imshow(torch.abs(out)[0, 0])
        
    elif mode =='1d_intensity':
        out = out[0, 0]
        H, W = out.shape
        out = out[H//2, :]
        x = np.linspace(-W//2, W//2, W, endpoint=True, dtype=np.int16)
        out_max = torch.abs(out.max())
        plt.plot(x, out/out_max)
    
    plt.title("_".join(this_key.split("_")[:4]))
    plt.tight_layout()        

def fresnel_compare_animate(i, dict_data, fresnel_dict_data, keys, mode='abs'):
    this_key = keys[i]
    out = dict_data[this_key]
    fresnel_out = fresnel_dict_data[this_key]
    plt.clf()
    
    if mode == 'abs':
        plt.subplot(121)
        plt.imshow(torch.abs(out)[0, 0])
        plt.title("_".join(this_key.split("_")[:3]))
        
        plt.subplot(122)
        plt.imshow(torch.abs(fresnel_out))
        plt.title("_".join(this_key.split("_")[:3]) + "_Fresnel")
     
        plt.tight_layout()
    
    elif mode =='1d_intensity':
        out = out[0, 0]
        H, W = out.shape
        FH, FW = fresnel_out.shape
        out = out[H//2, :]
        out_max = torch.abs(out.max())
        fresnel_out = fresnel_out[FH//2, :]      
        fresnel_out_max = torch.abs(fresnel_out.max())
        x = np.linspace(-W//2, W//2, W, endpoint=True, dtype=np.int16)
        plt.subplot(121)
        plt.plot(x, out/out_max)
        plt.title("_".join(this_key.split("_")[:3]))
        
        plt.subplot(122)
        plt.plot(x, fresnel_out/fresnel_out_max)
        plt.title("_".join(this_key.split("_")[:3]) + "_Fresnel")
        
        plt.tight_layout()
    

def f_num_iterative_perform(save_root, file_name, device):
    lamb0 = 0.55
    f_num_range = np.arange(0.1, 10.1, 0.1, dtype=np.float32)
    fixed_diameter = 100
    fixed_z = 10*1e3
    pupil_types = ['square']    
    this_file_name = "{}_{}_{}_" + file_name
    writer = animation.PillowWriter(fps=30)
    for pt in pupil_types:   
        kwargss = []
        # keys = []
        for f_num in f_num_range:
            #### fixed z
            f_num = round(f_num, 1)
            # print("This f_num: ", f_num)
            width = get_width(f_num, lamb=lamb0, z=fixed_z)
            kwargss.append(make_kwargs(lamb0, width, pt))
            # dict_key = f"fnum{round(f_num, 1)}_z{round(fixed_z, 1)}_d{round(width, 1)}_lamb0{lamb0}_{pt}pup"
            # keys.append(dict_key)
        if pt == 'square':
            var_w_out_dict, var_w_fresnel = iterative_perform_(kwargss, device, True)
            abs_this_animate = partial(fresnel_compare_animate, dict_data=var_w_out_dict, fresnel_dict_data=var_w_fresnel, keys=list(var_w_out_dict.keys()), mode='abs')
            int1d_this_animate = partial(fresnel_compare_animate, dict_data=var_w_out_dict, fresnel_dict_data=var_w_fresnel, keys=list(var_w_out_dict.keys()), mode='1d_intensity')
        else:
            var_w_out_dict = iterative_perform_(kwargss, device, False)
            abs_this_animate = partial(animate, dict_data=var_w_out_dict, keys=list(var_w_out_dict.keys()), mode='abs')
            int1d_this_animate = partial(animate, dict_data=var_w_out_dict, keys=list(var_w_out_dict.keys()), mode='1d_intensity')
            
        fig = plt.figure()
        ani = FuncAnimation(plt.gcf(), abs_this_animate, frames=len(kwargss), interval=100)
        ani.save(os.path.join(save_root, this_file_name.format(pt, 'abs', 'varwidth')), writer=writer)
        plt.close(fig)
        
        fig = plt.figure()
        ani = FuncAnimation(plt.gcf(), int1d_this_animate, frames=len(kwargss), interval=100)
        ani.save(os.path.join(save_root, this_file_name.format(pt, 'int', 'varwidth')), writer=writer)
        plt.close(fig)
        
        kwargss = []
        ### Video Visualization
        for f_num in f_num_range: 
            f_num = round(f_num, 1) 
            z = get_z(f_num, width=fixed_diameter, lamb=lamb0)
            kwargss.append(make_kwargs(lamb0, fixed_diameter, pt, z))
            # dict_key = f"fnum{round(f_num, 1)}_z{round(z, 1)}_d{round(fixed_diameter, 1)}_lamb0{lamb0}_{pt}pup"
            # keys.append(dict_key)
        if pt == 'square':
            var_z_out_dict, var_z_fresnel = iterative_perform_(kwargss, device, True)
            abs_this_animate = partial(fresnel_compare_animate, dict_data=var_z_out_dict, fresnel_dict_data=var_z_fresnel, keys=list(var_z_out_dict.keys()), mode='abs')
            int1d_this_animate = partial(fresnel_compare_animate, dict_data=var_z_out_dict, fresnel_dict_data=var_z_fresnel, keys=list(var_z_out_dict.keys()), mode='1d_intensity')
        else:
            var_z_out_dict = iterative_perform_(kwargss, device, False)
            abs_this_animate = partial(animate, dict_data=var_z_out_dict, keys=list(var_z_out_dict.keys()), mode='abs')
            int1d_this_animate = partial(animate, dict_data=var_z_out_dict, keys=list(var_z_out_dict.keys()), mode='1d_intensity')
        ### Video Visualization
        
        fig = plt.figure()
        ani = FuncAnimation(plt.gcf(), abs_this_animate, frames=len(kwargss), interval=100)
        ani.save(os.path.join(save_root, this_file_name.format(pt, 'abs', 'varz')), writer=writer)
        plt.close(fig)
        
        fig = plt.figure()
        ani = FuncAnimation(plt.gcf(), int1d_this_animate, frames=len(kwargss), interval=100)
        ani.save(os.path.join(save_root, this_file_name.format(pt, 'int', 'varz')), writer=writer)
        plt.close(fig)
        
        
if __name__ == "__main__":
    device = 'cuda:7'
    
    base_save_root = "./phase_prop_vis/diffraction"
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
    for pupil_type in ['square', 'circle']:
        for device in ['cuda:7', 'cpu']:
            save_root = os.path.join(base_save_root, pupil_type)
            os.makedirs(save_root, exist_ok=True)
            Prop = Diffraction(
                    pixel_size=[0.6, 0.6],
                    pixel_num=[1000, 1000],
                    lamb0=[0.4, 0.55, 0.7],
                    refractive_index=1,
                    z=10*1e3,
                    NA=0.3,
                    pupil_type=pupil_type,
                    pupil_width=100,
                    nyquist_spatial_bound=False
            ).to(device)
            strt = time.time()
            src_field, pupiled_field, prop_field, out = Prop()   
            end = time.time()
            
            print(f"{device} device : {end-strt} (s)")
        x_grid, y_grid = Prop.x_grid, Prop.y_grid
        radial_grid = x_grid ** 2 + y_grid ** 2
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
        
    iterative_perform(base_save_root, 'wvl_pupwidth_grid_fig.png', device=device) 
    
    f_num_iterative_perform(base_save_root, 'fnum_fig.gif', device=device)