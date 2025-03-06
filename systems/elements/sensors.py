import os, sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import torch
import torch.nn as nn
from systems.systems import Field



class Sensor(nn.Module):
    def __init__(self, shot_noise_modes, clip=(1e-20, 1.), channel_sum=True):
        super(Sensor, self).__init__()
        self.shot_noise_modes = shot_noise_modes
        self.clip = clip
        self.channel_sum = channel_sum
        
    def forward(self, field):
        return sensor(field, self.shot_noise_modes, self.clip, self.channel_sum)
        


def sensor(field: Field, shot_noise_modes: list, clip: list, channel_sum: bool = True):
    if channel_sum:
        intensity = field.intensity
    else:
        intensity = field.ch_intensity
    
    if isinstance(shot_noise_modes, str):
        shot_noise_modes = list(shot_noise_modes)
    
    for sn in shot_noise_modes:
        if sn == "gaussian":
            intensity = gaussian_noise(intensity)
        elif sn == "poisson":
            # intensity = torch.clamp(intensity, clip[0], 100.)
            intensity = poisson_noise(intensity)
        elif sn == "approximated_poisson":
            # intensity = torch.clamp(intensity, clip[0], 100.)
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