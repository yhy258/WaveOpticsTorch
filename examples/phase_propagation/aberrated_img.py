"""
!!! ISSUE:
    The simulated polychromatic PSF (independently simulated for several wavelengths) shows different scaling statistics
    for different wavelengths.
    -> How can we adjust and alleviate this problem??
    Green channel has the highest scale, Blue and Red channel have relatively lower scale.
    In my opinion, this problem may be caused by diffraction / defocusing issue..
    Is it okay if we use the raw PSF when simulating images?
"""


import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)
import torch
import torch.nn.functional as F
import systems.elements as elem
from systems.systems import OpticalSystem, Field
from systems.utils import _pair

def nyquist_pixelsize_criterion(NA, lamb):
    max_pixel_size = lamb/(2*NA) # C, Tensor
    return torch.min(max_pixel_size).item() # bound..

def global_normalize_abs(field):
    max_ch = torch.max(field.flatten())
    min_ch = torch.min(field.flatten())
    return (field - min_ch) / (max_ch - min_ch)

def normalize_abs(field):
    # field : H, W, C
    max_ch = torch.max(torch.max(field, dim=0, keepdim=True).values, dim=1, keepdim=True).values
    min_ch = torch.min(torch.min(field, dim=0, keepdim=True).values, dim=1, keepdim=True).values
    return (field - min_ch) / (max_ch - min_ch)

class AberratedImg(OpticalSystem):
    def __init__(
        self,
        pixel_size=[1, 1],
        pixel_num=[1000, 1000],
        lamb0=[0.455, 0.541, 0.616],
        target_lamb0=0.532,
        defocus=[], 
        refractive_index=1,
        paraxial=False,
        focal_length=19*1e3,
        NA=0.3,
        nyquist_spatial_bound=True,
    ):
        super(AberratedImg, self).__init__(
            pixel_size=pixel_size,
            pixel_num=pixel_num,
            lamb0=lamb0,
            refractive_index=refractive_index
        )
        defocus = torch.tensor(defocus)
        focal_length = torch.tensor(focal_length)
        
        max_pixel_size = nyquist_pixelsize_criterion(NA, self.lamb0/self.refractive_index)
        print("Max Pixel Size : ", max_pixel_size)
        print("Now Pixel Size : ", pixel_size)
        if max(pixel_size) > max_pixel_size and nyquist_spatial_bound:
            pixel_size = _pair(max_pixel_size)
            self.pixel_size = pixel_size
            self.init_grid_params()
        
        ### Collimated wave from point source. [ASSUMPTION]
        self.source = elem.PlaneSource(
            amplitude=1.0,
            ref_idx=self.refractive_index,
            power=1.0,
            dir_factors=None
        )
        width = min(pixel_num[0] * pixel_size[0], pixel_num[1] * pixel_size[1])
        self.circle = elem.CirclePupil(width=width)
        # 이 circle pupil이 들어간 이유는, digital 상의 grid의 크기가 lens diameter보다 훨씬 작아서 
        # 이 circle pupil을 취하지 않으면 사각형 형태의 PSF가 나오게됨.
        
        self.hyperboloid = elem.PhaseHyperboloid(pixel_num[0], pixel_num[1], refractive_idx=refractive_index, lamb0=target_lamb0, NA=NA, focal_length=focal_length, init_type='zeros')
        
        self.prop = elem.ASMPropagation(
            z=focal_length,
            ref_idx=self.refractive_index,
            band_limited=True
        )
        
        self.sensor = elem.Sensor(shot_noise_modes=[], clip=[1e-20, 1e+9], channel_sum=False)    
    
    def forward(self):
        field = Field(lamb0=self.lamb0, x_grid=self.x_grid, y_grid=self.y_grid, fx_grid=self.fx_grid, fy_grid=self.fy_grid)
        
        src_field = self.source(field)
        H, W = src_field.shape[-2:]
        src_field = self.circle(src_field)
        print(f"Initial Field's shape: {src_field.shape}")
        multp_field = self.hyperboloid(src_field)
        
        print(f"Field's shape after lens: {multp_field.shape}")    
        prop_field = self.prop(multp_field) # asm
        if isinstance(prop_field, list) or isinstance(prop_field, tuple): # SASPropagation.
            prop_field, pixel_size = prop_field
        print(f"Field's shape after propagation: {prop_field.shape}")
        out = self.sensor(prop_field)
        print(f"Output Field's shape : {out.shape}")
        return src_field.field, multp_field.field, None, prop_field.field, out
    


if __name__ == "__main__":
    
    device = 'cuda:3'
    #### This setting is the DRMI paper parameters. (approximated values)
    Prop = AberratedImg(
            pixel_size=[0.5, 0.5],
            pixel_num=[1000, 1000],
            lamb0=[0.450, 0.532, 0.635],
            target_lamb0=0.532,
            defocus=[29*1e3, 24.5*1e3, 20.5*1e3],
            refractive_index=1,
            paraxial=False,
            focal_length=24.5*1e3,
            NA=0.2,
            nyquist_spatial_bound=True
    ).to(device)
    src_field, multp_field, focus_errored_field, prop_field, out = Prop()   
    
    x_grid, y_grid = Prop.x_grid, Prop.y_grid
    radial_grid = x_grid ** 2 + y_grid ** 2
    
    import matplotlib.pyplot as plt
    save_root = "./phase_prop_vis/aberrated_img"
    os.makedirs(save_root, exist_ok=True)
    lamb0 = Prop.lamb0 # list
    ### visualize function
    def visualize(file_name, field, title, mode='phase', multch_normalize=False):
        if field.device != 'cpu':
            field = field.detach().cpu()
        if mode == "abs":
            if multch_normalize:
                # field = normalize_abs(field)
                field = global_normalize_abs(field)
            plt.imshow(torch.abs(field))    
            
        elif mode == "phase" or mode == 'angle':
            plt.imshow(torch.angle(field))
        elif mode == "real":
            plt.imshow(field.real)
        plt.title(title + "_" + mode)
        plt.colorbar()
        plt.savefig(os.path.join(save_root, file_name))
        plt.clf()
        
    visualize("Radial Grid", radial_grid, "Radial GRID x**2 + y**2", mode='abs')
    
    file_name_format = "{}_field_{}.png"
    title_format = "{} field of Lambda : {}"
    
    visualize(file_name_format.format('multp_field', torch.round(lamb0[0]/Prop.nanometers)), multp_field[0, 0], title=title_format.format("multp_field", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('prop_field', torch.round(lamb0[0]/Prop.nanometers)), prop_field[0, 0], title=title_format.format("prop_field", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
    
    visualize(file_name_format.format('multp_field', torch.round(lamb0[1]/Prop.nanometers)), multp_field[0, 1], title=title_format.format("multp_field", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('prop_field', torch.round(lamb0[1]/Prop.nanometers)), prop_field[0, 1], title=title_format.format("prop_field", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
    
    visualize(file_name_format.format('multp_field', torch.round(lamb0[2]/Prop.nanometers)), multp_field[0, 2], title=title_format.format("multp_field", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('prop_field', torch.round(lamb0[2]/Prop.nanometers)), prop_field[0, 2], title=title_format.format("prop_field", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')

    visualize(file_name_format.format('source', torch.round(lamb0[0]/Prop.nanometers)), src_field[0, 0], title=title_format.format("Source", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('source', torch.round(lamb0[1]/Prop.nanometers)), src_field[0, 1], title=title_format.format("Source", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('source', torch.round(lamb0[2]/Prop.nanometers)), src_field[0, 2], title=title_format.format("Source", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')
    
    visualize(file_name_format.format('sensor', torch.round(lamb0[0]/Prop.nanometers)), out[0, 0], title=title_format.format("Sensor", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('sensor', torch.round(lamb0[1]/Prop.nanometers)), out[0, 1], title=title_format.format("Sensor", torch.round(lamb0[1]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('sensor', torch.round(lamb0[2]/Prop.nanometers)), out[0, 2], title=title_format.format("Sensor", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
    
    ##### 적절한 scaling 필요... - Calibration.
    visualize(file_name_format.format('sensor', 'Multichannel'), out[0].permute(1, 2, 0), title=title_format.format("Sensor", 'Multichannel'), mode='abs', multch_normalize=True)
    
