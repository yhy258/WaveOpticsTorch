import math
import numpy as np
import torch
import torch.nn.functional as F


NP_DTYPE = np.complex64
T_DTYPE = torch.float32


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
    real = input.cos().unsqueeze(-1)
    imag = input.sin().unsqueeze(-1)
    return real + 1j*imag


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


# This should be edited.. -> complex computation does not require additional dimension.
# a is sample, and b is psf.
def fftconv2d(a, b, fa=None, fb=None, shape='same', fftsize=None):
    # sample : B, C, H, W or B, C, D, H, W = a
    # psf : D, N, N = b
    
    asize = a.size()
    bsize = b.size()
    
    if fftsize == None:
        # 두개 합친 크기
        fftsize = [asize[-2] + bsize[-2] - 1, asize[-1] + bsize[-1] - 1]
        
    # a : B, C, H, W or B, C, D, H, W
    if fa is None:
        
        fa = torch.fft.fftn(
            F.pad(
                a, (0, fftsize[-1] - asize[-1], 0, fftsize[-2] - asize[-2])
            ), dim=[-2, -1]
        ) # B, ..., H + alpha, W + alpha ; complex value.
        # fa = torch.fft(
        #     F.pad(
        #         a.unsqueeze(-1),
        #         (0, 1, 0, fftsize[-1] - asize[-1], 0, fftsize[-2] - asize[-2])
        #     ), # B, ..., H+alpha, W+alpha, 2
        #     2,
        # )
        
    if fb is None:
        fb = torch.fft.fftn(
            F.pad(
                b, (0, fftsize[-1] - bsize[-1], 0, fftsize[-2] - bsize[-2])
            ), dim=[-2, -1]
        ) # B, ..., H + alpha, W + alpha ; complex value.
        fb = filter_IEEE_error(F.pad(
                b, (0, fftsize[-1] - bsize[-1], 0, fftsize[-2] - bsize[-2])
            ), fb)
        # fb = torch.fft(
        #     F.pad(
        #         b.unsqueeze(-1),
        #         (0, 1, 0, fftsize[-1] - bsize[-1], 0, fftsize[-2] - bsize[-2])
        #     ), # B, ..., H+alpha, W+alpha, 2
        #     2,
        # )
        
    ab = torch.fft.ifftn(fa * fb, dim=(-2, -1))
    # ab = (
    #     torch.fft.ifftn(fa * fb, dim=(-2, -1))
    #     .index_select(-1, torch.tensor(0, device=fa.device))
    #     .squeeze(-1)
    # )
    
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