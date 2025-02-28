import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
from utils import compute_intensity



class Sensor(nn.Module):
    def __init__(self, shot_noise_modes, clip=(1e-20, 1.)):
        super(Sensor, self).__init__()
        self.shot_noise_modes = shot_noise_modes
        self.clip = clip
        
    def forward(self, field):
        return sensor(field, self.shot_noise_modes, self.clip)
        


def sensor(field, shot_noise_modes, clip):
    
    intensity = compute_intensity(field)
    
    if isinstance(shot_noise_modes, str):
        shot_noises = list(shot_noises)
    
    for sn in shot_noises:
        if sn == "gaussian":
            intensity = gaussian_noise(intensity)
        elif sn == "poisson":
            field = torch.clamp(field, clip[0], 100.)
            intensity = poisson_noise(intensity)
        elif sn == "approximated_poisson":
            field = torch.clamp(field, clip[0], 100.)
            intensity = approximated_poisson_noise(intensity)
    
    return torch.clamp(intensity, clip[0], clip[1])



def poisson_noise(image, a=1.):
    return torch.poisson(image/a)

def approximated_poisson_noise(image):
    eps = torch.randn_like(image)
    approx_mu = image + torch.sqrt(image) * eps
    return torch.maximum(approx_mu, torch.zeros_like(approx_mu))

def gaussian_noise(image, std):
    return image + torch.randn_like(image) * std