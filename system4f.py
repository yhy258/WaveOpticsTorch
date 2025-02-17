"""
    This code is basically based on "https://github.com/TuragaLab/snapshotscope"
"""
# TODO : Convert the complex computation -> inherent datatype..!

import numpy as np
import scipy
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

NP_DTYPE = np.complex64
T_DTYPE = torch.float32
TC_DTYPE = torch.complex64

### UTILS
def _list(x, repetitions=1):
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return x
    else:
        return [
            x,
        ] * repetitions



class Pupil4FLensSystem(nn.Module):
    def __init__(
        self,
        wavelength: list = [0.532],
        ratios: list = [1.0],
        NA: float = 0.8,
        ref_idx: float = 1.33,
        pixel_size: float = 0.325,
        num_pixels: int = 2560,
        pad: int = 0,
        taper_width: float = 0,
        downsample: int = 1,
        device = 'cpu' 
    ):
        super(Pupil4FLensSystem, self).__init__()
        
        ### Wavelength should be defined with List. We can allocate multiple wavelengths (chromatic case)
        self.wavelength = _list(wavelength)
        self.ratios = _list(ratios) # linear combination coeffs of each wavelength case.
        self.NA = NA
        self.ref_idx = ref_idx
        self.pixel_size = pixel_size
        self.num_pixels = num_pixels + pad 
        self.pad = pad
        self.taper_width = taper_width
        self.downsample = downsample
        self.device = device
        
        self.dx = self.pixel_size
        self.fS = 1 / self.dx # spatial sampling frequency
        # spacing between discrete frequency coordinates, inverse microns        
        # The smallest sinusoidal component that can exist in this plane has delta x * N size.
        # That means, the smallest frequency component ; 1/ (delta x * N) = dk
        # frequency : 1 / entire size.
        """
            If we want to set k as wavevector, we have to change this term
            self.dk = 2pi / (self.pixel_size * self.num_pixels)
        """
        self.dk = 1 / (self.pixel_size * self.num_pixels)
        self.fNA = [self.NA / w for w in self.wavelength] # NA/lambda <- maximum value of kx, ky.
        
        
        # image plane coord
        self.x = ( 
            np.linspace(
                -(self.num_pixels - 1) / 2,
                (self.num_pixels - 1) / 2,
                num=self.num_pixels,
                endpoint=True
            ) * self.dx
        )
        
        self.xy_range = [self.x[0], self.x[-1]]
        self.psf_size = [self.num_pixels, self.num_pixels] # N by N. This is just freq domain..
        self.fourier_psf_size = [
            2 * self.num_pixels - 1,
            2 * self.num_pixels - 1,
            2,
        ]
        
        if self.downsample > 1:
            self.downsampled_psf_size = [
                int(d / self.downsample) for d in self.psf_size
            ]
            self.downsampled_fourier_psf_size = [
                2 * self.downsampled_psf_size[0] - 1,
                2 * self.downsampled_psf_size[1] - 1,
                2,
            ]
            
        # Fourier plane.
        self.k = (
            np.linspace(
                -(self.num_pixels - 1)/2,
                (self.num_pixels - 1)/2, 
                num=self.num_pixels,
                endpoint=True
            ) * self.dk
        ) 
        self.k_range = [self.k[0], self.k[-1]]
        kx, ky = np.meshgrid(self.k, self.k) # (N x N)
        self.register_buffer('kx', torch.tensor(kx, dtype=T_DTYPE, deivce=device))
        self.register_buffer('ky', torch.tensor(ky, dtype=T_DTYPE, device=device))
        
        # create the pupil
        """
            If we want to set k as wavevector, we have to change this term
            pupil_mask = [1.0 * 1/(2*pi) * (np.sqrt(kx ** 2 + ky ** 2) <= self.fNA[i]) for i in range(len(self.wavelength))]
        """
        pupil_mask = [
            1.0 * (np.sqrt(kx ** 2 + ky ** 2) <= self.fNA[i])
            for i in range(len(self.wavelength))
        ] # consider chromatic case. L * (N x N), where L is the number of considered wavelengths.
        # (The number of on-mask components * self.dk)**2
        pupil_power = [(m**2).sum() * self.dk ** 2 for m in pupil_mask]
        pupil = [
            pupil_mask[i] / np.sqrt(pupil_power[i]) for i in range(len(self.wavelength))
        ] # L * (N x N)
        pupil = np.stack(pupil) # L x N x N
        self.register_buffer("pupil", torch.tensor(pupil, dtype=TC_DTYPE, device=self.device))
        # self.register_buffer("pupil", ctensor_from_numpy(pupil, device=self.device))
        
        
        """
            If we want to set k as wavevector, we have to change this term
            np.sqrt(
                ((2*np.pi * self.ref_idx / self.wavelength[i]) ** 2 - (kx ** 2 + ky ** 2)) * pupil_mask[i]
            )
            pupil_mask = [1.0 * 1/(2*pi) * (np.sqrt(kx ** 2 + ky ** 2) <= self.fNA[i]) for i in range(len(self.wavelength))]
            
            
            # This defocus phase angle is developed with the following processes.
            1. Propagation of diverging spherical source (d = f) using paraxial approx
            2. Multiplicative phase transformation of converging lens using paraxial approx
            3. ASM ; assumption : lens pupil func is a unity and ignoring the coef term 
        """
        # 2pi * z * asm. 형태인데, z가 빠져있음 -> 이는 depth에 따라 나중에 다르게 부여할 예정. (defocus_z)
        defocus_phase_angle = [
            2 * np.pi * np.sqrt(
                ((self.ref_idx / self.wavelength[i]) ** 2 - (kx ** 2 + ky ** 2)) * pupil_mask[i]
            )
            for i in range(len(self.wavelength))
        ]
        
        defocus_phase_angle = np.stack(defocus_phase_angle) # L x N x N
        
        self.register_buffer(
            "defocus_phase_angle",
            torch.tensor(defocus_phase_angle, dtype=T_DTYPE, device=self.device),
        )
        
        if self.pad > 0:
            distance_input = np.pad(
                np.ones(
                    (
                        self.num_pixels - self.pad - 2,
                        self.num_pixels - self.pad - 2,
                    )
                ),
                1,
            )
            
            distance = scipy.ndimage.distance_transform_edt(distance_input)
            # taper function. taper width denotes how quickly the taper goes to 0 at the edges.
            self.taper = 2 * (
                torch.sigmoid(torch.from_numpy(distance) / self.taper_width) - 0.5
            ).to(self.device)
    
    # complex component is allocated in additional dimension
    def compute_psf(self, phase_mask, defocus_z):
        # calculate psf
        
        pupils = torch.split(self.pupil, 1) # [pupil] per lambda
        defocus_phase_angles = torch.split(self.defocus_phase_angle, 1) # incidenct field on pupil plane.
        psf = 0
        
        for i in range(len(self.wavelength)):
            # pupil_phase
            # pupils[i].squeeze(0) : N x N, phase_mask : N x N (all types are complex64)
            pupil_phase = pupils[i].squeeze(0) * phase_mask * self.wavelength[0] / self.wavelength[i]
            # pupil_phase = cmul(
            #     pupils[i].squeeze(0),
            #     phase_mask * self.wavelength[0] / self.wavelength[i]
            # ) # relative wavelength, N x N x 2
            
            # TODO: WHat is da defocus_z? -> phase.cos + jphase.sin 
            # maybe depth에 따른 phase difference인듯?
            defocus = ctensor_from_phase_angle(
                defocus_phase_angles[i].squeeze(0).mul(defocus_z)
            ) # N x N 
            
            pupil_phase_defocus = pupil_phase * defocus # N x N
            # pupil_phase_defocus = cmul(pupil_phase, defocus) # multiplication between incidenct field and pupil function
            
            # from colimating field to f - f lens system, -> just fft.
            
            im_field = (self.dk ** 2) * torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(pupil_phase_defocus))
            ).unsqueeze(0) # 1 x N x N
            # self.dk ** 2 for normalizing the discrepency between dx dy vs dk dk..
            
            
            # im_field = (self.dk ** 2) * fftshift2d(
            #     torch.fft(ifftshift2d(pupil_phase_defocus), 2)
            # ).unsqueeze(0)
            # PSF : I
            weighted_im_field = (self.ratios[i] * torch.abs(im_field).pow()).squeeze() # N x N
            # weighted_im_field = (self.ratios[i] * cabs2(im_field)).squeeze()
            psf += weighted_im_field.squeeze()
        return psf # N x N
        
    def image(self, phase_mask_angle, sample, defocus_zs):
        phase_mask = ctensor_from_phase_angle(phase_mask_angle) # N x N
        im = 0
        psf = []
        
        for zidx, defocus_z in enumerate(defocus_zs): # D
            psf.append(self.compute_psf(phase_mask, defocus_z))
        psf = torch.stack(psf)
        
        self.psf = psf
        
        crop = int(self.pad / 2)
        if crop > 0:
            psf = psf[:, crop:-crop, crop:-crop]
            psf = psf * self.taper.expand_as(psf)
            self.cropped_psf = psf
        if self.downsample > 1:
            psf = F.avg_pool2d(
                psf.unsqueeze(1),
                kernel_size=self.downsample,
                divisor_override=1
            ).squeeze(1)
        # sample : B, C, H, W or B, C, D, H, W
        # psf : D, N, N
        im = fftconv2d(sample, psf, shape='same').sum(0)
        return im
        