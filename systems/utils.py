from itertools import repeat
from functools import partial
import math
import numpy as np
import torch
import torch.nn.functional as F


NP_DTYPE = np.complex64
T_DTYPE = torch.float32


def _list(x, repetitions=1):
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return x
    else:
        return [
            x,
        ] * repetitions


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

def set_spatial_grid(H, W, dx, dy):
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, (H-1), H) - H / 2,
        np.linspace(0, (W-1), W) - W / 2, indexing='ij'
    )
    grid = torch.stack((torch.from_numpy(grid_x)*dx, torch.from_numpy(grid_y)*dy), dim=0)
    grid.requires_grad_(False) 
    return grid # Tensor (2, H, W)

def set_freq_grid(H, W, dx, dy):
    fx, fy = np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(H, dx)), 
        np.fft.fftshift(np.fft.fftfreq(W, dy)), indexing='ij'
    )
    f_grid = torch.stack((torch.from_numpy(fx), torch.from_numpy(fy)), dim=0)
    f_grid.requires_grad_(False) 
    return f_grid
    

### MATH
def combinatorial(n, k):
    """Calculates the combination C(n, k), i.e. n choose k."""
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

def pearson_corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    corr = (
        torch.sum(x * y)
        * torch.rsqrt(torch.sum(x ** 2))
        * torch.rsqrt(torch.sum(y ** 2))
    )
    return corr


### COMPLEX OPERATIONS
def ctensor_from_numpy(input, dtype=T_DTYPE, device="cpu"):
    real = torch.tensor(input.real, dtype=dtype, device=device).unsqueeze(-1)
    imag = torch.tensor(input.imag, dtype=dtype, device=device).unsqueeze(-1)
    output = torch.cat([real, imag], np.ndim(input))
    return output

def ctensor_from_phase_angle(input):
    return torch.exp(1j * input)

def compute_intensity(field, sum=True):
    # field : B, C, H, W -> return : B, 1, H, W
    if sum:
        return torch.sum(torch.abs(field) ** 2, dim=1, keepdim=True)
    else:
        return torch.abs(field) ** 2

def compute_power(field, pixel_area):
    intensity = compute_intensity(field)
    return torch.sum(intensity, dim=(-2, -1), keepdim=True) * pixel_area # B, 1, 1, 1

### FOURIER OPERATIONS

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

def double_padnd(x, n=2):
    xsize = x.size()[::-1][:n]
    fftsize = [xsz*2 - 1 for xsz in xsize]
    padsize = ()
    for fsz, xsz in zip(fftsize, xsize):
        padsize += (0, fsz - xsz)
    return F.pad(x, padsize)

def double_unpadnd(x, n=2):
    fxsize = x.size()[::-1][:n]
    ifftsize = [(xsz+1)//2 for xsz in fxsize]
    unpadsize = ()
    for ifsz, fxsz in zip(ifftsize, fxsize):
        unpadsize += (0, -abs(ifsz-fxsz))
    x = F.pad(x, unpadsize)
    return x
    
def padded_fftnd(x, n=2):
    transform_dims = -np.arange(1, n+1)[::-1].tolist()
    x = double_padnd(x, n=n)
    fx = torch.fft.fftn(x, dim=transform_dims)
    return fx

def unpadded_ifftnd(fx, n=2):
    transform_dims = (-np.arange(1, n+1)[::-1]).tolist()
    out = torch.fft.ifftn(fx, dim=transform_dims)
    x = double_unpadnd(out, n=n)
    return x
    
###### CONVOLUTION THEOREM
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

###### https://github.com/chromatix-team/chromatix/blob/main/src/chromatix/functional/convenience.py#L8
###### Optical fft for 2F lens system... (Input is the collimated light)


def shifted_fft(x, spatial_dims=(-2, -1)):
    fft = partial(torch.fft.fft2, dim=spatial_dims)
    ifftshift = partial(torch.fft.ifftshift, dim=spatial_dims)
    fftshift = partial(torch.fft.fftshift, dim=spatial_dims)
    return fftshift(fft(ifftshift(x)))


def optical_fft(field, grid, f_grid, lamb0, z, ref_idx):
    """
    Args:
        field (Tensor, B, C, H, W): Input field of this system.
        grid (Tensor, 2, H, W): Spatial grid
        f_grid (Tensor, 2, H, W): Frequency grid
        lamb0 (Tensor, C,): WAvelengths
        z (float): Propagation distance
        ref_idx (Int): Refractive Index
        
        # but only consider forward propagation.
    """

    L_sq = lamb0[None, :, None, None] * z / ref_idx # 1, C, 1, 1
    
    dx = torch.abs(grid[0, 1, 0] - grid[0, 0, 0])
    dk = torch.abs(f_grid[0, 1, 0] - f_grid[0, 0, 0])
    du = dk * L_sq
    
    norm_fft = -1j * dx * dx / L_sq # B, C, 1, 1
    
    fft_input = norm_fft * field # B, C, H, W
    
    spatial_dims = (-2, -1)
    
    fft_output = shifted_fft(fft_input, spatial_dims=spatial_dims)
    
    return fft_output
 


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




def gaussian_kernel_3d(
    sigma=1.0, kernel_size=(21, 21, 21), pixel_size=1.0, device="cpu"
):
    """Creates a 3D Gaussian kernel of specified size and pixel size."""
    kernel_size = _triple(kernel_size)
    sigma = _triple(sigma)
    z, y, x = torch.meshgrid(
        [
            torch.linspace(
                -(kernel_size[d] - 1) / 2,
                (kernel_size[d] - 1) / 2,
                steps=kernel_size[d],
            )
            * pixel_size
            for d in range(3)
        ]
    )
    kernel = 1
    for d, s in zip([z, y, x], sigma):
        d_mean = d[tuple(int(d.shape[i] / 2) for i in range(3))]
        kernel *= (1 / (s * math.sqrt(2 * math.pi))) * torch.exp(
            -(((d - d_mean) / (2 * s)) ** 2)
        )
    kernel = kernel / torch.sum(kernel)
    return kernel.to(device)


def gaussian_blur_3d(
    vol3d,
    sigma=1.0,
    kernel_size=(21, 21, 21),
    pixel_size=1.0,
    kernel=None,
    device="cpu",
):
    """Performs 3D blur using the specified kernel and FFT convolution."""
    if kernel is None:
        kernel = gaussian_kernel_3d(sigma, kernel_size, pixel_size, device)
    return fftconvnd(vol3d, kernel, n=3, shape="same")


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

_triple = _ntuple(3)
_pair = _ntuple(2)
_single = _ntuple(1)

# #%%
# import numpy as np

# x1 = np.linspace(1, 5, 5)
# y1 = np.linspace(6, 10, 5)

# x2, y2 = np.meshgrid(x1, y1, indexing='ij')
# print(x1, y1)
# # %%
# x2, y2
# %%
