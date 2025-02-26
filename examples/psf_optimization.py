import os, sys, math, datetime, glob, faulthandler
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from system4f import Pupil4FLensSystem
from optical_elements import *
from core.models import *

import logging
from examples import opt4fsys_train_psf as ctrl
from examples import dataloaders as dl
from examples import augment


logging.basicConfig(filename='out.log', level=logging.DEBUG, format="%(message)s")

learning_rate = 1e-4
pm_learning_rate = 1e2 * learning_rate # phase mask learning rate
regularize_lost_power = 0
num_iterations = 1000000


# setup the simulation parameters, make a 4f system.
num_pixels = 2560
pixel_size = 0.325
num_systems = 2 # for parallelization
num_planes = 256 
downsample = 5
plane_subsample = 4
psf_pad = 1280
taper_width = 5
regularize_power = False
devices = [torch.device(f'cuda:{i}') for i in range(num_systems)]

# define SLM Parameters
circle_NA = 0.8
pupil_NA = 0.8
wavelength = 0.532
fNA = circle_NA / wavelength
slm_radius_pixels = 678.374 * circle_NA
slm_num_pixels = int(slm_radius_pixels * 2) + 1
"""
The difference between dk in slm and dk in Fourier plane (4f lens system)
1. slm's dk : slm plays a role in modulating phase. -> the maximum frequency it can modulate is NA/lambda.
    The interval is determined by the resolution of slm (the number of pixels)
    slm_dk = fNA / slm_radius_pixels
    ; 즉, SLM은 phase를 modulating 하기 위한 장치임. 그리고 SLM은 Fourier space에 위치하여있고, spatial frequency component를 조절함.
    ; 시스템의 parameter에 대해 maximum spatial frequency가 정해짐.
    ; 즉, SLM이 Fourier space에 위치해있을떄, SLM pixels의 infinitesimal은 maximum_freq / num_pixels로 정해지는거임.
2. dk in Fourier domain : it is actually the Fourier space corresponding to physical space
    The frequency is highly related to the sampling rate in physical space (inverse relationship)
    The infinitesimal in Fourier space would be 1 / (pixel_size * num_pixels)
    ; pixel * num_pixels로 정의된 공간이 있다면, 존재할 수 있는 가장 큰 sinusoidal function의 period는 pixel * num_pixels.
    ; 즉, 가장 낮은 frequency는 1/pixel_size * num_pixels가 되는거임.
"""
slm_dk = fNA / slm_radius_pixels
slm_k = (
    np.linspace(
        -slm_num_pixels//2 +1,
        slm_num_pixels//2 +1,
        num=slm_num_pixels,
        endpoint=True,
        dtype=np.float32
    ) * slm_dk
)
slm_k = torch.from_numpy(slm_k)
slm_kx, slm_ky = torch.meshgrid(slm_k, slm_k)

# calculate downsampled sizes
subsampled_num_planes = int(num_planes / plane_subsample)
downsampled_num_pixels = int(num_pixels / downsample)
downsampled_radius = int(0.5 * 386 / pixel_size / downsample) # maybe, 386 is diameter, and downsampled_radius is the number of pixels for radius

# set grad sizes
# TODO: 이게 뭐지.
num_grad_im_planes = 5
num_grad_recon_planes = 5

# create chunked defocus ranges
defocus_range = np.round(np.linspace(-125, 125, num=subsampled_num_planes))
chunk_size = int(len(defocus_range) / num_systems) # the size of allocated depth per an optical system instance.
if len(defocus_range) % num_systems != 0:
    chunk_size += 1 
sections = [i * chunk_size for i in range(1, num_systems)]
defocus_ranges = np.split(defocus_range, sections)

# create 4fsystem param dicts
param_dict = dict(
    wavelength = [wavelength],
    ratios = [1.0],
    NA = pupil_NA,
    ref_idx = 1.33,
    pixel_size = pixel_size,
    num_pixels = num_pixels,
    pad = psf_pad,
    taper_width = taper_width,
    downsample = downsample
)

param_dicts = [
    dict(list(param_dict.items()) + [('device', device)]) for device in devices
]

## TODO: augmentation info
# create augmentations
augmentations = augment.create_augmentations(
    [
        {
            "name": "adjust_brightness",
            "args": {
                "scale": 52.8,
                "background": (0, 60),
                "brightness": (0.33, 3.0),
                "log": False,
            },
        },
        {"name": "rotate_pitch", "args": {"angle": (-5, 5)}},
        {"name": "rotate_roll", "args": {"angle": (-10, 10)}},
        {"name": "rotate_yaw", "args": {"angle": (-20, 20)}},
        {
            "name": "horizontal_jitter",
            "args": {"jitter_amount": (0, 60), "direction": ("left", "right")},
        },
        {
            "name": "flip_planes",
            "args": {"allowed_flips": [0, 2], "num_flips": "random"},
        },
        {"name": "flip_volume", "args": {"probability": 0.5}},
        {"name": "reflect_planes", "args": {"probability": 0.5}},
        {
            "name": "cylinder_crop",
            "args": {
                "radius": downsampled_radius,
                "center": "center",
                "max_depth": subsampled_num_planes,
            },
        },
        {"name": "slice_crop", "args": {"num_planes": int(subsampled_num_planes / 10)}},
        {
            "name": "pad_volume",
            "args": {
                "target_shape": (
                    subsampled_num_planes,
                    downsampled_num_pixels,
                    downsampled_num_pixels,
                )
            },
        },
    ]
)
augmentations = augment.compose_augmentations(augmentations)


test_augmentations = augment.create_augmentations(
    [
        {
            "name": "adjust_brightness",
            "args": {"scale": 52.8, "background": 0, "brightness": 1, "log": False},
        },
        {"name": "slice_crop", "args": {"num_planes": int(subsampled_num_planes / 10)}},
        {
            "name": "pad_volume",
            "args": {
                "target_shape": (
                    subsampled_num_planes,
                    downsampled_num_pixels * 2,
                    downsampled_num_pixels,
                )
            },
        },
    ]
)
test_augmentations = augment.compose_augmentations(test_augmentations)


def create_4fsystems():
    opts = [Pupil4FLensSystem(**param_dict) for param_dict in param_dicts]
    opts = nn.ModuleList(opts)
    return opts

def create_phase_mask(kx, ky, phase_mask_init=None):
    if phase_mask_init is None:
        # TODO: defocusing + phase ramping. -> Maybe I should study this.
        defocused_ramps = DefocusedRamps(
            kx, ky, pupil_NA / wavelength, 1.33, wavelength, delta=2374.0
        )
        phase_mask_init = defocused_ramps()
    pixels = Pixels(kx, ky, pixels=phase_mask_init)
    return pixels

def create_dataloader(test=False):
    base_path = '/data/joon/Dataset/FourierNetData/LarvalJebrafish/interpolated_1.625um'
    # if not test:
    if test == False:
        dataset = dl.ConfocalVolumesDataset(
            [
                os.path.join(base_path, "zjabc_train/2019-12-10-6dpf-Huc-H2B-jRGECO"),
                os.path.join(base_path, "zjabc_train/2019-12-16-5dpf-Huc-H2B-G7FF"),
                os.path.join(base_path, "zjabc_train/2019-12-17-6dpf-Huc-H2B-G7FF"),
                os.path.join(
                    base_path,
                    "zjabc_train/2020-02-25_abc-Elavl3-H2B-GCaMP+Cytosolic-GCaMP",
                ),
                os.path.join(base_path, "zjabc_train/2020-02-27_abc-Elavl3-H2B-GCaMP"),
                os.path.join(
                    base_path, "zjabc_train/2020-02-28_abc-Elval3-H2B-GCaMP+jRGECO"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-01-5dpf-elavl3-H2B-jRGECO"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-02-6dpf-elavl3-H2B-jRGECO"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-03_abc-Elavl3-H2B-GC7f_gfapjRGECO1b"
                ),
                os.path.join(base_path, "zjabc_train/2020-03-03-Elavl3-H2B-GCaMP"),
                os.path.join(
                    base_path, "zjabc_train/2020-03-15-_Elavl3-H2B-GC7ff_gfapjRGECO1b"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-17-Elavl3-H2B-GC7f_gfapjRGECO1b"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-18-Elavl3-H2B-GC7f_gfapjRGECO1b"
                ),
            ],
            shape=(num_planes, downsampled_num_pixels, downsampled_num_pixels),
            location="random",
            strategy="center",
            bounds_roi="data",
            valid_ratio=(0.5, 1.0, 0.1),
            steps=(plane_subsample, 1, 1),
            augmentations=augmentations,
            balance=True,
        )
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, shuffle=True)
    else:
        dataset = dl.ConfocalVolumesDataset(
            [
                os.path.join(
                    base_path, "zjabc_test/2019-02-24_abc-Elavl3-H2B-GC7f_gfapjRGECO1b"
                ),
                os.path.join(base_path, "zjabc_test/2020-03-04-5dpf-elva3-H2B-jRGECO"),
            ],
            shape=(num_planes, downsampled_num_pixels * 2, downsampled_num_pixels),
            location="center",
            strategy="center",
            bounds_roi="data",
            steps=(plane_subsample, 1, 1),
            augmentations=test_augmentations,
            balance=False,
        )
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, shuffle=False)
    return dataloader

def create_reconstruction_networks():
    deconvs = []
    planes_per_device = int(subsampled_num_planes / num_systems) # device별 planes.
    for device in devices:
        device_deconvs = [
            FourierNet2D(
                5, # the number of output channels of Fourier Layer.
                (downsampled_num_pixels, downsampled_num_pixels),
                1,
                fourier_conv_args={"stride" : 2},
                conv_kernel_sizes=[11],
                input_scaling_mode="scaling",
                quantile=0.5,
                quantile_scale=1e-2,
                device="cpu"
            )
            for p in range(planes_per_device)
        ]
        deconvs.extend(device_deconvs)
    if len(deconvs) < subsampled_num_planes:
        device_deconvs = [
            FourierNet2D(
                5,
                (downsampled_num_pixels, downsampled_num_pixels),
                1,
                fourier_conv_args={"stride": 2},
                conv_kernel_sizes=[11],
                input_scaling_mode="scaling",
                quantile=0.5,
                quantile_scale=1e-2,
                device="cpu",
            )
            for p in range(subsampled_num_planes - len(deconvs))
        ]
        deconvs.extend(device_deconvs)
    deconvs = nn.ModuleList(deconvs)
    return deconvs


def create_placeholder_reconstruction_networks():
    # create multi gpu reconstruction network
    deconvs = []
    planes_per_device = int(subsampled_num_planes / num_systems)
    for device in devices:
        device_deconvs = [
            FourierNet2D(
                5,
                (downsampled_num_pixels, downsampled_num_pixels),
                1,
                fourier_conv_args={"stride": 2},
                conv_kernel_sizes=[11],
                input_scaling_mode="scaling",
                quantile=0.5,
                quantile_scale=1e-2,
                device=device, # device 별로 할당.
            )
            for p in range(num_grad_recon_planes)
        ]
        deconvs.append(nn.ModuleList(device_deconvs))
    deconvs = nn.ModuleList(deconvs)
    return deconvs

def initialize_optimizer(optdeconv, latest=None):
    ### This should be replaced with custom RAdam to use the params as dict form.
    opt = optim.RAdam(
        [
            {
                "params": optdeconv['placeholder_deconvs'].parameters(),
                "lr": learning_rate,
            },
            {
                "params": optdeconv["phase_mask"].parameters(), "lr": pm_learning_rate
            }
        ]
    )
    if latest is not None:
        opt.load_state_dict(latest["opt_state_dict"])
    return opt


def initialize_4fsystem_reconstruction(latest=None, phase_mask_init=None):
    opts = create_4fsystems()
    pixels = create_phase_mask(opts[0].kx, opts[0].ky, phase_mask_init=phase_mask_init)
    deconvs = create_reconstruction_networks()
    placeholder_deconvs = create_placeholder_reconstruction_networks()
    optdeconv = nn.ModuleDict(
        {
            "opts": opts,
            "deconvs": deconvs,
            "placeholder_deconvs": placeholder_deconvs,
            "phase_mask": pixels
        }
    )
    if latest is not None:
        print("[info] loading from checkpoint")
        optdeconv.load_state_dict(latest["optdeconv_state_dict"], strict=True)
    return optdeconv

def create_test_extents():
    # define extents as list of lists of tuples of tuples
    # len(extents) = number of samples in dataloader
    # each item in extents is a list of locations in the sample
    # each location is a tuple of dimensions
    # each dimension is a tuple containing a start and end index
    length = downsampled_radius * 2
    extents = [
        [
            ((0, subsampled_num_planes), (280, 280 + length), (120, 120 + length)),
            ((0, subsampled_num_planes), (600, 600 + length), (138, 138 + length)),
        ],
        [
            ((0, subsampled_num_planes), (280, 280 + length), (138, 138 + length)),
            ((0, subsampled_num_planes), (550, 550 + length), (138, 138 + length)),
        ],
        [
            ((0, subsampled_num_planes), (190, 190 + length), (138, 138 + length)),
            ((0, subsampled_num_planes), (400, 400 + length), (138, 138 + length)),
        ],
        [
            ((0, subsampled_num_planes), (200, 200 + length), (138, 138 + length)),
            ((0, subsampled_num_planes), (500, 500 + length), (150, 150 + length)),
        ],
        [
            ((0, subsampled_num_planes), (155, 155 + length), (120, 120 + length)),
            ((0, subsampled_num_planes), (400, 400 + length), (150, 150 + length)),
        ],
        [
            ((0, subsampled_num_planes), (190, 190 + length), (138, 138 + length)),
            ((0, subsampled_num_planes), (460, 460 + length), (138, 138 + length)),
        ],
        [
            ((0, subsampled_num_planes), (230, 230 + length), (138, 138 + length)),
            ((0, subsampled_num_planes), (520, 520 + length), (138, 138 + length)),
        ],
        [
            ((0, subsampled_num_planes), (230, 230 + length), (138, 138 + length)),
            ((0, subsampled_num_planes), (520, 520 + length), (138, 138 + length)),
        ],
        [
            ((0, subsampled_num_planes), (190, 190 + length), (138, 138 + length)),
            ((0, subsampled_num_planes), (460, 460 + length), (138, 138 + length)),
        ],
        [
            ((0, subsampled_num_planes), (240, 240 + length), (120, 120 + length)),
            ((0, subsampled_num_planes), (460, 460 + length), (138, 138 + length)),
        ],
    ]
    return extents
def train():
    
    # initialize model for training.
    if os.path.exists("latest.pt"):
        latest = torch.load("latest.pt")
    else:
        latest = None
        
        
    optdeconv = initialize_4fsystem_reconstruction(latest=latest)
    print(optdeconv)
    
    # initialize optimizer
    optimizer = initialize_optimizer(optdeconv, latest=latest)


    # initialize data
    dataloader = create_dataloader()
    val_dataloader = create_dataloader(test=True)
    
    #initialize test extents
    extents = create_test_extents()
    
    # initialize iteration count
    if latest is not None:
        latest_iter = latest["it"]
        losses = latest["mses"]
        high_pass_losses = latest["high_pass_mses"]
        regularized_losses = latest["regularized_losses"]
        validate_losses = latest["validate_high_pass_mses"]
    else:
        latest_iter = 0
        losses = []
        high_pass_losses = []
        regularized_losses = []
        validate_losses = []
        
    if latest is not None:
        del latest
        torch.cuda.empty_cache()
        
        
    # create mse loss
    mse = nn.MSELoss()

    # initialize iteration count
    it = int(latest_iter)

    # create folder for validation data
    if not os.path.exists("snapshots/validate/"):
        os.makedirs("snapshots/validate/", exist_ok=True)
    val_dir = "snapshots/validate/"
    

    ctrl.train_psf(
        optdeconv,
        optimizer,
        dataloader,
        defocus_range,
        num_grad_im_planes,
        num_grad_recon_planes,
        devices,
        losses,
        high_pass_losses,
        regularized_losses,
        num_iterations,
        single_decoder=False,
        input_3d=False,
        regularize_lost_power=regularize_lost_power,
        high_pass_kernel_size=11,
        low_pass_weight=0.1,
        validate_losses=validate_losses,
        validate_args={
            "dataloader": val_dataloader,
            "defocus_range": defocus_range,
            "target_shape": (
                subsampled_num_planes,
                downsampled_num_pixels,
                downsampled_num_pixels,
            ),
            "devices": devices,
            "extents": extents,
            "num_grad_recons_planes": num_grad_recon_planes,
            "single_decoder": False,
            "input_3d": False,
            "high_pass_kernel_size": 11,
            "aperture_radius": downsampled_radius,
            "save_dir": val_dir,
        },
        it=it
    )
    
    
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(3, num_systems+3)))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,5"
    train()