from itertools import repeat

import math
import numpy as np
import torch
import torch.nn.functional as F


NP_DTYPE = np.complex64
T_DTYPE = torch.float32


### CHECKPOINT UTILITIES
def copy_params_(source, target):
    for s, t in zip(source.parameters(), target.parameters()):
        t.data.copy_(s.data)

def copy_deconv_params_to_placeholder(num_systems, optdeconv, recon_chunk_zidxs):
    for didx in range(num_systems):
        for cidx, zidx in enumerate(recon_chunk_zidxs[didx]):
            copy_params_(
                optdeconv["deconvs"][zidx], optdeconv["placeholder_deconvs"][didx][cidx]
            )

def copy_placeholder_params_to_deconv(num_systems, optdeconv, recon_chunk_zidxs):
    for didx in range(num_systems):
        for cidx, zidx in enumerate(recon_chunk_zidxs[didx]):
            copy_params_(
                optdeconv["placeholder_deconvs"][didx][cidx], optdeconv["deconvs"][zidx]   
            )
    

### COORDINATE SYSTEM
def cart2pol(x, y):
    rho = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(y, x)
    return (rho, theta)


def pol2cart(rho, theta):
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    return x, y


### MATH
def combinatorial(n, k):
    """Calculates the combination C(n, k), i.e. n choose k."""
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

### COMPLEX OPERATIONS

# def cabs2(x):
#     xabs2 = x.pow(2).sum(-1)
#     return xabs2

def ctensor_from_numpy(input, dtype=T_DTYPE, device="cpu"):
    real = torch.tensor(input.real, dtype=dtype, device=device).unsqueeze(-1)
    imag = torch.tensor(input.imag, dtype=dtype, device=device).unsqueeze(-1)
    output = torch.cat([real, imag], np.ndim(input))
    return output

# def ctensor_from_phase_angle(input):
#     real = input.cos().unsqueeze(-1)
#     imag = input.sin().unsqueeze(-1)
#     output = torch.cat([real, imag], input.ndimension())
#     return output

def ctensor_from_phase_angle(input):
    return torch.exp(1j * input)
    


# def cmul(x, y, xy=None):
#     if xy is None:
#         xy = torch.zeros_like(x)
#     x2 = x.view(-1, 2)
#     y2 = y.view(-1, 2)
#     xy2 = xy.view(-1, 2)
#     xy2[:, 0] = torch.mul(x2[:, 0], y2[:, 0]) - torch.mul(x2[:, 1], y2[:, 1])
#     xy2[:, 1] = torch.mul(x2[:, 0], y2[:, 1]) + torch.mul(x2[:, 1], y2[:, 0])
#     return xy


### FOURIER OPERATIONS


# def fftshift(input, dim=1):
#     split_size_right = int(np.floor(input.size(dim) / 2))
#     split_sizes = (input.size(dim) - split_size_right, split_size_right)
#     pos, neg = torch.split(input, split_sizes, dim=dim)
#     input = torch.cat([neg, pos], dim=dim)
#     return input


# def ifftshift(input, dim=1):
#     split_size_left = int(np.floor(input.size(dim) / 2))
#     split_sizes = (split_size_left, input.size(dim) - split_size_left)
#     pos, neg = torch.split(input, split_sizes, dim=dim)
#     input = torch.cat([neg, pos], dim=dim)
#     return input


# def fftshift2d(input):
#     for dim in [0, 1]:
#         input = fftshift(input, dim=dim)
#     return input


# def ifftshift2d(input):
#     for dim in [0, 1]:
#         input = ifftshift(input, dim=dim)
#     return input


def filter_IEEE_error(psf, otf):
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    # n_ops denotes n log n
    # the below means it the integration (Fourier ..) outputs are smaller than nlogn * IEEE 754 eps, filter it as 0.
    # otf[..., 1][torch.abs(otf.real) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    threshold = n_ops*2.22e-16
    otf = torch.complex(
        torch.where(torch.abs(otf.real) < threshold, torch.tensor(0., dtype=otf.real.dtype, device=otf.device), otf.real),
        torch.where(torch.abs(otf.imag) < threshold, torch.tensor(0., dtype=otf.imag.dtype, device=otf.device), otf.imag)
    )
    return otf


# a is sample, and b is psf.
# Successfully verified this (The use of rfft and irfft)
def fftconv2d(a, b, fa=None, fb=None, shape='same', fftsize=None):
    # sample : B, C, H, W or B, C, D, H, W = a
    # psf : D, N, N = b
    
    asize = a.size()
    bsize = b.size()
    
    if fftsize == None:
        # The added size.
        fftsize = [asize[-2] + bsize[-2] - 1, asize[-1] + bsize[-1] - 1]
        
    # a : B, C, H, W or B, C, D, H, W
    if fa is None:
        
        fa = torch.fft.rfft2(
            F.pad(
                a, (0, fftsize[-1] - asize[-1], 0, fftsize[-2] - asize[-2])
            ), dim=[-2, -1]
        )
    if fb is None:
        paded_b = F.pad(
            b, (0, fftsize[-1] - bsize[-1], 0, fftsize[-2] - bsize[-2])
        )
    
        fb = torch.fft.rfft2(paded_b)
        
        # fb = filter_IEEE_error(F.pad(
        #         b, (0, fftsize[-1] - bsize[-1], 0, fftsize[-2] - bsize[-2])
        #     ), fb)
        fb = filter_IEEE_error(paded_b, fb)
    ab = torch.fft.irfft2(fa * fb, dim=(-2, -1), s=fftsize)
    
    
    # crop based on shape
    if shape in "same":
        cropsize = [fftsize[-2] - asize[-2], fftsize[-1] - asize[-1]] # PSF size
        cropsizeL = [int(c / 2) for c in cropsize]
        cropsizeR = [int((c + 1) / 2) for c in cropsize]
        ab = F.pad(
            ab,
            (-cropsizeL[-1], -cropsizeR[-1], -cropsizeL[-2], -cropsizeR[-2]),
        )
    elif shape in "valid":
        cropsize = [
            fftsize[-2] - asize[-2] + bsize[-2] - 1,
            fftsize[-1] - asize[-1] + bsize[-1] - 1,
        ]
        cropsizeL = [int(c / 2) for c in cropsize]
        cropsizeR = [int(c / 2) for c in cropsize]
        ab = F.pad(
            ab,
            (-cropsizeL[-1], -cropsizeR[-1], -cropsizeL[-2], -cropsizeR[-2]),
        )
        
    return ab

# a is sample, and b is psf.
# Successfully verified this (The use of rfft and irfft)
def fftconvnd(a, b, n=3, fa=None, fb=None, shape='same', fftsize=None):
    # sample : B, C, H, W or B, C, D, H, W = a
    # psf : D, N, N = b
    transform_dims = -np.arange(1, n+1)[::-1].tolist()
    asize = a.size()[::-1][:n]
    bsize = b.size()[::-1][:n]
    
    if fftsize == None:
        # The added size.
        fftsize = [asz + bsz - 1 for asz, bsz in zip(asize, bsize)]
        
    # a : B, C, H, W or B, C, D, H, W
    if fa is None:
        padsize = ()
        for fsz, asz in zip(fftsize, asize):
            padsize += (0, fsz - asz)
        fa = torch.fft.rfftn(F.pad(a, padsize), dim=transform_dims)

    if fb is None:
        padsize = ()
        for fsz, bsz in zip(fftsize, bsize):
            padsize += (0, fsz - bsz)
        
        paded_b = F.pad(b, padsize)
    
        fb = torch.fft.rfftn(paded_b, dim=transform_dims)
        
        fb = filter_IEEE_error(paded_b, fb)
        
    ab = torch.fft.irfftn(fa * fb, dim=transform_dims, s=fftsize)
    
    
    # crop based on shape
    if shape in "same":
        cropsize = [fsz - asz for fsz, asz in zip(fftsize, asize)]
        padsize = ()
        for c in cropsize:
            padsize += (-int(c / 2), -int((c + 1) / 2))
        ab = F.pad(ab, padsize)
    elif shape in "valid":
        cropsize = [fsz - asz + bsz - 1 for fsz, asz, bsz in zip(fftsize, asize, bsize)]
        padsize = ()
        for c in cropsize:
            padsize += (-int(c / 2), -int((c + 1) / 2))
        ab = F.pad(ab, padsize)

    return ab



### IMAGING UTILS
def gaussian_kernel_2d(sigma, kernel_size, device="cpu"):
    kernel_size = _pair(kernel_size)
    y, x = torch.meshgrid(
        (
            torch.arange(kernel_size[0]).float(),
            torch.arange(kernel_size[1]).float(),
        )
    )
    y_mean = (kernel_size[0] - 1) / 2.0
    x_mean = (kernel_size[1] - 1) / 2.0
    kernel = 1
    kernel *= (
        1
        / (sigma * math.sqrt(2 * math.pi))
        * torch.exp(-(((y - y_mean) / (2 * sigma)) ** 2))
    )
    kernel *= (
        1
        / (sigma * math.sqrt(2 * math.pi))
        * torch.exp(-(((x - x_mean) / (2 * sigma)) ** 2))
    )
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.to(device)
    return kernel


def high_pass_filter(vol, high_pass_kernel, use_3d=False):
    if use_3d:
        return fftconvnd(vol, high_pass_kernel, n=3, shape="same")
    else:
        return fftconv2d(vol, high_pass_kernel, shape="same") # return fftconvnd(vol, high_pass_kernel, n=2, shape="same") ??


def poissonlike_gaussian_noise(im):
    im = im + (im.sqrt()) * torch.randn_like(im)
    im.clamp_(min=0.0)
    return im


def _ntuple(n):
    """Creates a function enforcing ``x`` to be a tuple of ``n`` elements."""

    def parse(x):
        if isinstance(x, tuple):
            return x
        return tuple(repeat(x, n))

    return parse

_pair = _ntuple(2)
_single = _ntuple(1)