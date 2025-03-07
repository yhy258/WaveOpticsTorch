import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import systems.elements as elem
from systems.systems import OpticalSystem, Field
from systems.utils import _pair
import time

### Verification of Utilization of GPU devices.

def nyquist_pixelsize_criterion(NA, lamb):
    max_pixel_size = lamb/(2*NA) # C, Tensor
    return torch.min(max_pixel_size).item() # bound..
    

class SpatialFilteredPSF(OpticalSystem):
    def __init__(
        self,
        pixel_size=[1, 1],
        pixel_num=[1000, 1000],
        lamb0=[0.55, 1.05, 1.550],
        refractive_index=1,
        paraxial=False,
        focal_length=19*1e3,
        NA=0.3,
        pinhole_width=20,
        nyquist_spatial_bound=True,
    ):
        super(SpatialFilteredPSF, self).__init__(
            pixel_size=pixel_size,
            pixel_num=pixel_num,
            lamb0=lamb0,
            refractive_index=refractive_index
        )
        
        
        max_pixel_size = nyquist_pixelsize_criterion(NA, self.lamb0/self.refractive_index)
        print("Max Pixel Size : ", max_pixel_size)
        print("Now Pixel Size : ", pixel_size)
        if max(pixel_size) > max_pixel_size and nyquist_spatial_bound:
            pixel_size = _pair(max_pixel_size)
            self.pixel_size = pixel_size
            self.init_grid_params()
        
        self.source = elem.PointSource(
            amplitude=1.0,
            ref_idx=self.refractive_index,
            z=focal_length,
            src_loc=None, # center.
            power=1.0,
            paraxial=paraxial
        )
        
        self.lens = elem.MultConvergingLens(
            ref_idx=self.refractive_index,
            focal_length=focal_length,
            NA=NA
        ) # including pupil func
        
        self.prop = elem.ASMPropagation(
            z=focal_length,
            ref_idx=self.refractive_index,
            band_limited=True
        )
        
        self.register_buffer("pinhole", elem.circular_pupil(self.x_grid, self.y_grid, pinhole_width))
        
        self.fflens = elem.FFLens(
            ref_idx=self.refractive_index,
            focal_length=focal_length,
            NA=NA
        )
        
        self.sensor = elem.Sensor(shot_noise_modes=[], clip=[1e-20, 1e+9], channel_sum=False)    
    
    def forward(self):
        #####
        """
            Wavelength, FGrid, Grid, Field values should be tracked during forward process rather than initialization process!!!!
            Field class
            1. Plus, Minus, Element-wise multip
            2. Device Allocation
            3. 
        """
        
        field = Field(lamb0=self.lamb0, x_grid=self.x_grid, y_grid=self.y_grid, fx_grid=self.fx_grid, fy_grid=self.fy_grid)
        
        strt = time.time()
        src_field = self.source(field)
        print( type(src_field.field))

        end = time.time()
        process_time = end-strt
        print(f"Initial Field's shape: {src_field.shape}\tProcess Time: {process_time}")
    
        
        strt = time.time()
        multp_field = self.lens(src_field) # lens
        end = time.time()
        process_time = end-strt
        print(f"Field's shape after lens: {multp_field.shape}\tProcess Time: {process_time}")    
        
        strt = time.time()
        prop_field = self.prop(multp_field) # asm
        end = time.time()
        process_time = end-strt
        if isinstance(prop_field, list) or isinstance(prop_field, tuple): # SASPropagation.
            prop_field, pixel_size = prop_field
        print(f"Field's shape after propagation: {prop_field.shape}\tProcess Time: {process_time}")
        
        strt = time.time()
        pinhole_field = prop_field * self.pinhole[None, None, :, :]
        end = time.time()
        process_time = end-strt
        print(f"Field's shape after pinhole: {pinhole_field.shape}\tProcess Time: {process_time}") # spatial filtering
        
        strt =time.time()
        ff_field = self.fflens(pinhole_field)
        end = time.time()
        process_time = end-strt
        print(f"FF lens Field's shape : {ff_field.shape}\tProcess Time: {process_time}")
        
        strt = time.time()
        out = self.sensor(ff_field)
        end = time.time()
        process_time = end-strt
        print(f"Output Field's shape : {out.shape}\tProcess Time: {process_time}")
        return src_field.field, multp_field.field, prop_field.field, pinhole_field.field, ff_field.field, out
    
    

def make_kwargs(lamb0, NA):
    this_kwargs = dict(
        pixel_size=[0.5, 0.5],
        pixel_num=[300, 300],
        lamb0=[lamb0],
        refractive_index=1,
        paraxial=False,
        focal_length=1*1e3,
        NA=NA,
        pinhole_width=100,
        nyquist_spatial_bound=False
    )
    return this_kwargs
    
def iterative_perform_(kwargss: list):
    outs = {}
    for kwargs in kwargss:
        Prop = SpatialFilteredPSF(**kwargs).to('cuda:7')
        src_field, multp_field, prop_field, pinhole_field, ff_field, out = Prop()
        #### Visualization code.
        #### Out field instantly save in list.
        #### Show the figures in grid!
        out = out.detach().cpu()
        NA = kwargs['NA']
        lamb0 = kwargs['lamb0']
        
        dict_key = f"NA{NA}_lamb0{lamb0[0]}"
        outs[dict_key] = out
    return outs

def iterative_perform(save_root, file_name):
    lamb0s = [0.4, 0.55, 0.7]
    NAs = [0.1, 0.3, 0.5]
    kwargss = []
    for lamb0 in lamb0s:
        for NA in NAs:
            kwargss.append(make_kwargs(lamb0, NA))
    out_dict = iterative_perform_(kwargss)
    ### Visualization with varying out_dict[f"NA{NA}_lamb0{lamb0}"]
    ### Make Fig grid
    fig, axes = plt.subplots(nrows=len(lamb0s), ncols=len(NAs))
    
    for i, lamb0 in enumerate(lamb0s):
        for j, NA in enumerate(NAs):
            dict_key = f"NA{NA}_lamb0{lamb0}"
            out = out_dict[dict_key]
            axes[i,j].imshow(torch.abs(out)[0,0])
            axes[i,j].title.set_text(dict_key) 
            
    plt.tight_layout()
    fig.savefig(os.path.join(save_root, file_name))
    plt.clf()


if __name__ == "__main__":
    Prop = SpatialFilteredPSF(
            pixel_size=[5, 5],
            pixel_num=[1000, 1000],
            lamb0=[0.55, 1.05, 1.550],
            refractive_index=1,
            paraxial=False,
            focal_length=19*1e3,
            NA=0.3,
            pinhole_width=20,
            nyquist_spatial_bound=False
    ).to('cuda:7')
    src_field, multp_field, prop_field, pinhole_field, ff_field, out = Prop()   
    src_field, multp_field, prop_field, pinhole_field, ff_field, out  = src_field.detach().cpu(), multp_field.detach().cpu(), prop_field.detach().cpu(), pinhole_field.detach().cpu(), ff_field.detach().cpu(), out.detach().cpu() 
    import matplotlib.pyplot as plt
    save_root = "./phase_prop_vis/spatial_fltr_no_pinhole"
    os.makedirs(save_root, exist_ok=True)
    lamb0 = Prop.lamb0 # list
    x_grid, y_grid = Prop.x_grid, Prop.y_grid # list
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
        
    visualize("Radial Grid", x_grid ** 2 + y_grid ** 2, "Radial GRID x**2 + y**2", mode='abs')
    
    file_name_format = "{}_field_{}.png"
    title_format = "{} field of Lambda : {}"
    
    visualize(file_name_format.format('multp_field', torch.round(lamb0[0]/Prop.nanometers)), multp_field[0, 0], title=title_format.format("multp_field", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('prop_field', torch.round(lamb0[0]/Prop.nanometers)), prop_field[0, 0], title=title_format.format("prop_field", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('pinhole_field', torch.round(lamb0[0]/Prop.nanometers)), pinhole_field[0, 0], title=title_format.format("pinhole_field", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('ff_field', torch.round(lamb0[0]/Prop.nanometers)), ff_field[0, 0], title=title_format.format("ff_field", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    
    visualize(file_name_format.format('multp_field', torch.round(lamb0[2]/Prop.nanometers)), multp_field[0, 2], title=title_format.format("multp_field", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('prop_field', torch.round(lamb0[2]/Prop.nanometers)), prop_field[0, 2], title=title_format.format("prop_field", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('pinhole_field', torch.round(lamb0[2]/Prop.nanometers)), pinhole_field[0, 2], title=title_format.format("pinhole_field", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('ff_field', torch.round(lamb0[2]/Prop.nanometers)), ff_field[0, 2], title=title_format.format("ff_field", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
    
    
    visualize(file_name_format.format('source', torch.round(lamb0[0]/Prop.nanometers)), src_field[0, 0], title=title_format.format("Source", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('source', torch.round(lamb0[1]/Prop.nanometers)), src_field[0, 1], title=title_format.format("Source", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('source', torch.round(lamb0[2]/Prop.nanometers)), src_field[0, 2], title=title_format.format("Source", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')
    
    visualize(file_name_format.format('sensor', torch.round(lamb0[0]/Prop.nanometers)), out[0, 0], title=title_format.format("Sensor", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('sensor', torch.round(lamb0[1]/Prop.nanometers)), out[0, 1], title=title_format.format("Sensor", torch.round(lamb0[1]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('sensor', torch.round(lamb0[2]/Prop.nanometers)), out[0, 2], title=title_format.format("Sensor", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
    
    iterative_perform(save_root, 'wvl_NA_grid_fig.png')