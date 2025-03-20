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
from systems.utils import _pair, D2NA

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



"""
    This class is the streamlined version of the original imaging system ; https://www.spiedigitallibrary.org/journals/advanced-photonics/volume-6/issue-06/066002/Deep-learning-driven-end-to-end-metalens-imaging/10.1117/1.AP.6.6.066002.full
    I thought this simplified model could emulate the original model...
    But, when I matched this class's parameters with the original imaging system's parameters
    there are discrepancies in PSFs' tendencies.
"""
# class AberratedImg(OpticalSystem):
#     def __init__(
#         self,
#         pixel_size=[1, 1],
#         pixel_num=[1000, 1000],
#         lamb0=[0.455, 0.541, 0.616],
#         target_lamb0=0.532,
#         defocus=[], 
#         refractive_index=1,
#         paraxial=False,
#         focal_length=24.5*1e3,
#         D=10*1e3,
#         nyquist_spatial_bound=True,
#     ):
#         super(AberratedImg, self).__init__(
#             pixel_size=pixel_size,
#             pixel_num=pixel_num,
#             lamb0=lamb0,
#             refractive_index=refractive_index
#         )
#         defocus = torch.tensor(defocus)
#         focal_length = torch.tensor(focal_length)
#         NA = D2NA(D, focal_length, refractive_index)
#         max_pixel_size = nyquist_pixelsize_criterion(NA, self.lamb0/self.refractive_index)
#         print("Max Pixel Size : ", max_pixel_size)
#         print("Now Pixel Size : ", pixel_size)
#         if max(pixel_size) > max_pixel_size and nyquist_spatial_bound:
#             pixel_size = _pair(max_pixel_size)
#             self.pixel_size = pixel_size
#             self.init_grid_params()
        
#         ### Collimated wave from point source. [ASSUMPTION]
#         self.source = elem.PlaneSource(
#             amplitude=1.0,
#             ref_idx=self.refractive_index,
#             power=1.0,
#             dir_factors=None
#         )
#         width = min(pixel_num[0] * pixel_size[0], pixel_num[1] * pixel_size[1])
#         self.circle = elem.CirclePupil(width=width)
        
#         ### THIS EMULATES THE METALENS.
#         self.hyperboloid = elem.PhaseHyperboloid(pixel_num[0], pixel_num[1], refractive_idx=refractive_index, lamb0=target_lamb0, D=D, focal_length=focal_length, init_type='zeros')
        
#         self.prop = elem.ASMPropagation(
#             z=focal_length,
#             ref_idx=self.refractive_index,
#             band_limited=True
#         )
        
#         self.sensor = elem.Sensor(shot_noise_modes=[], clip=[1e-20, 1e+9], channel_sum=False)    
    
#     def forward(self):
#         field = Field(lamb0=self.lamb0, x_grid=self.x_grid, y_grid=self.y_grid, fx_grid=self.fx_grid, fy_grid=self.fy_grid)
        
#         src_field = self.source(field)
#         H, W = src_field.shape[-2:]
#         src_field = self.circle(src_field)
#         print(f"Initial Field's shape: {src_field.shape}")
#         multp_field = self.hyperboloid(src_field)
        
#         print(f"Field's shape after lens: {multp_field.shape}")    
#         prop_field = self.prop(multp_field) # asm
#         if isinstance(prop_field, list) or isinstance(prop_field, tuple): # SASPropagation.
#             prop_field, pixel_size = prop_field
#         print(f"Field's shape after propagation: {prop_field.shape}")
#         out = self.sensor(prop_field)
#         print(f"Output Field's shape : {out.shape}")
#         return src_field.field, multp_field.field, None, prop_field.field, out

"""
    Spatially filtered PSF.
    Paper: https://www.spiedigitallibrary.org/journals/advanced-photonics/volume-6/issue-06/066002/Deep-learning-driven-end-to-end-metalens-imaging/10.1117/1.AP.6.6.066002.full
    1. Achromatic doublet(Thorlabs: AC127-019-A) - f = 19 mm (12.7mm diameter)
    2. 20µm pinhole
    3. Spherical Lens (Thorlabs: LA1461-A - N-BK7 Plano-Convex Lens) - f = 250 mm (25.4 mm diameter)
    ### Collimated.
    4. Phase modulation.. Note that this doesn't represent the phase modulation of the metalens in the paper.
    ### Just emulating the process.. If we train the learnable parameter in the phase mask module, then we may fit the phase profile with metalens
    ### But it is impossible to perfectly emulate the metalens's phase profile,  
        because the PSF imaging simulation hypothesize spatially invariant degradation and 
        this wave optics setting may make it hard to train the sub-wavelength characteristics of the metalens.
"""
class OGAberratedImg(OpticalSystem):
    def __init__(
        self,
        pixel_size=[1, 1],
        pixel_num=[1500, 1500],
        lamb0=[0.455, 0.541, 0.616],
        target_lamb0=0.532,
        defocus=[], 
        refractive_index=1,
        paraxial=False,
        focal_lengths=[19*1e3, 250*1e3, 24.5*1e3],
        Ds=[12.7*1e3, 25.4*1e3, 10*1e3], ### diameter를 감쌀 수 있을 정도의 grid를 형성시켜줘야하지 않은가?
        pinhole_width=20,
        nyquist_spatial_bound=True,
    ):
        super(OGAberratedImg, self).__init__(
            pixel_size=pixel_size,
            pixel_num=pixel_num,
            lamb0=lamb0,
            refractive_index=refractive_index
        )
        defocus = torch.tensor(defocus)
        focal_lengths = torch.tensor(focal_lengths)
        Ds = torch.tensor(Ds)
        NA = D2NA(Ds, focal_lengths, refractive_index)
        NA = torch.max(NA)
        max_pixel_size = nyquist_pixelsize_criterion(NA, self.lamb0/self.refractive_index)
        print("Max Pixel Size : ", max_pixel_size)
        print("Now Pixel Size : ", pixel_size)
        if max(pixel_size) > max_pixel_size and nyquist_spatial_bound:
            pixel_size = _pair(max_pixel_size)
            self.pixel_size = pixel_size
            self.init_grid_params()
        
        ### Collimated wave from point source. [ASSUMPTION]
        self.source = elem.PointSource(
            amplitude=1.0,
            ref_idx=self.refractive_index,
            power=1.0,
            z=focal_lengths[0],
            src_loc=None
        )
        width = min(pixel_num[0] * pixel_size[0], pixel_num[1] * pixel_size[1])
        self.circle = elem.CirclePupil(width=width)
        
        ### THIS EMULATES THE METALENS.
        self.lens1 = elem.MultConvergingLens(ref_idx=refractive_index, focal_length=focal_lengths[0], D=Ds[0])
        self.prop1 = elem.ASMPropagation(
            z=focal_lengths[0],
            ref_idx=self.refractive_index,
            band_limited=True
        )
        
        self.pinhole = elem.CirclePupil(width=pinhole_width)
        self.prop2 = elem.ASMPropagation(
            z=focal_lengths[1],
            ref_idx=self.refractive_index,
            band_limited=True
        )
        self.lens2 = elem.MultConvergingLens(ref_idx=refractive_index, focal_length=focal_lengths[1], D=Ds[1])
        # prop2
        
        """
            Hyperboloid만 사용해도 자연스럽게 chromatic aberration이 고려되는가?
        """
        self.hyperboloid = elem.PhaseHyperboloid(pixel_num[0], pixel_num[1], refractive_idx=refractive_index, lamb0=target_lamb0, D=Ds[2], focal_length=focal_lengths[2], init_type='zeros')
        
        self.prop3 = elem.ASMPropagation(
            z=focal_lengths[2],
            ref_idx=self.refractive_index,
            band_limited=True
        )
        
        self.sensor = elem.Sensor(shot_noise_modes=[], clip=[1e-20, 1e+9], channel_sum=False)    
    
    def forward(self):
        field = Field(lamb0=self.lamb0, x_grid=self.x_grid, y_grid=self.y_grid, fx_grid=self.fx_grid, fy_grid=self.fy_grid)
        
        src_field = self.source(field)
        H, W = src_field.shape[-2:]
        # src_field = self.circle(src_field)
        print(f"Initial Field's shape: {src_field.shape}")
        
        lens1_field = self.lens1(src_field)
        prop1_field = self.prop1(lens1_field)
        pinhole_field = self.pinhole(prop1_field)
        
        prop2_field = self.prop2(pinhole_field)
        lens2_field = self.lens2(prop2_field)
        prop2_2_field = self.prop2(lens2_field)
        
        mod_field = self.hyperboloid(prop2_2_field)
        prop3_field = self.prop3(mod_field)
        
        out = self.sensor(prop3_field)
        print(f"Output Field's shape : {out.shape}")
        return src_field.field, lens1_field.field, lens2_field.field, mod_field.field, out

if __name__ == "__main__":
    
    device = 'cuda'
    #### This setting is the DRMI paper parameters. (approximated values)
    # Prop = AberratedImg(
    #         pixel_size=[1, 1],
    #         pixel_num=[500, 500],
    #         lamb0=[0.450, 0.532, 0.635],
    #         target_lamb0=0.532,
    #         defocus=[29*1e3, 24.5*1e3, 20.5*1e3],
    #         refractive_index=1,
    #         paraxial=False,
    #         focal_length=24.5*1e3,
    #         D=10*1e3,
    #         nyquist_spatial_bound=True
    # ).to(device)
    
    """
        1. The grid size should be sufficiently large to cover the entire transverse propagation space.
        2. The transverse space of the propagation after pinhole is propotional to lambda and z, and inversely proportional to pinhole's width.
    """
    Prop = OGAberratedImg(
            pixel_size=[0.8, 0.8],
            pixel_num=[2000, 2000],
            lamb0=[0.450, 0.532, 0.635],
            target_lamb0=0.532,
            defocus=[29*1e3, 24.5*1e3, 20.5*1e3],
            refractive_index=1,
            paraxial=False,
            focal_lengths=[19*1e3, 19*1e3, 24.5*1e3],
            Ds=[10*1e3, 10*1e3, 10*1e3],
            pinhole_width=20,
            nyquist_spatial_bound=True
    ).to(device)
    src_field, lens1_field, lens2_field, mod_field, out = Prop()   
    
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
    
    visualize(file_name_format.format('lens1_field', torch.round(lamb0[0]/Prop.nanometers)), lens1_field[0, 0], title=title_format.format("lens1_field", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('lens2_field', torch.round(lamb0[0]/Prop.nanometers)), lens2_field[0, 0], title=title_format.format("lens2_field", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('modulated_field', torch.round(lamb0[0]/Prop.nanometers)), mod_field[0, 0], title=title_format.format("modulated_field", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
    
    visualize(file_name_format.format('lens1_field', torch.round(lamb0[1]/Prop.nanometers)), lens1_field[0, 1], title=title_format.format("lens1_field", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('lens2_field', torch.round(lamb0[1]/Prop.nanometers)), lens2_field[0, 1], title=title_format.format("lens2_field", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('modulated_field', torch.round(lamb0[1]/Prop.nanometers)), mod_field[0, 1], title=title_format.format("modulated_field", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
    
    visualize(file_name_format.format('lens1_field', torch.round(lamb0[2]/Prop.nanometers)), lens1_field[0, 2], title=title_format.format("lens1_field", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('lens2_field', torch.round(lamb0[2]/Prop.nanometers)), lens2_field[0, 2], title=title_format.format("lens2_field", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('modulated_field', torch.round(lamb0[2]/Prop.nanometers)), mod_field[0, 2], title=title_format.format("modulated_field", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')

    visualize(file_name_format.format('source', torch.round(lamb0[0]/Prop.nanometers)), src_field[0, 0], title=title_format.format("Source", torch.round(lamb0[0]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('source', torch.round(lamb0[1]/Prop.nanometers)), src_field[0, 1], title=title_format.format("Source", torch.round(lamb0[1]/Prop.nanometers)), mode='angle')
    visualize(file_name_format.format('source', torch.round(lamb0[2]/Prop.nanometers)), src_field[0, 2], title=title_format.format("Source", torch.round(lamb0[2]/Prop.nanometers)), mode='angle')
    
    visualize(file_name_format.format('sensor', torch.round(lamb0[0]/Prop.nanometers)), out[0, 0], title=title_format.format("Sensor", torch.round(lamb0[0]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('sensor', torch.round(lamb0[1]/Prop.nanometers)), out[0, 1], title=title_format.format("Sensor", torch.round(lamb0[1]/Prop.nanometers)), mode='abs')
    visualize(file_name_format.format('sensor', torch.round(lamb0[2]/Prop.nanometers)), out[0, 2], title=title_format.format("Sensor", torch.round(lamb0[2]/Prop.nanometers)), mode='abs')
    
    ##### 적절한 scaling 필요... - Calibration.
    visualize(file_name_format.format('sensor', 'Multichannel'), out[0].permute(1, 2, 0), title=title_format.format("Sensor", 'Multichannel'), mode='abs', multch_normalize=True)
    
