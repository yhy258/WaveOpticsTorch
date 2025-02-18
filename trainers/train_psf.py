import os, sys, math, datetime, glob, faulthandler
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
)
import logging
import time

import numpy as np
import torch
import torch.cuda.comm
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda._utils import _get_device_index
from utils import *
from parallel_utils import gather, select_chunk_indices, take_chunk_indices, chunk_indices_diff, selective_scatter, parallel_apply


def train_psf(
    optdeconv,
    optimizer,
    dataloader,
    defocus_range,
    num_grad_im_planes, # these are for sparse-gradient calculation.
    num_grad_recon_planes,
    devices,
    losses,
    high_pass_losses,
    regularized_losses,
    num_iterations,
    lr_scheduler=None,
    single_decoder=False,
    input_3d=False,
    im_function=None,
    regularize_lost_power=0,
    high_pass_kernel_size=11,
    low_pass_weight=0,
    validate_losses=[],
    validate_args=None,
    it=0,  
):
    """
    Trains a PSF for volume reconstruction given a potentially multi-GPU
    decoder and microscope architecture. Supports selective gradients for
    certain planes during imaging and reconstruction.
    """
    
    num_systems = len(optdeconv['opts'])
    
    #### CHUNK CONSTANTS FOR PARALLELIZATION.
    chunk_size = int(len(defocus_range) / num_systems)
    if len(defocus_range) % num_systems != 0:
        chunk_size += 1
    
    ## If len(defocus_range) cannot be divided with num_systems.. -> remainder factor exists.
    # chunk_size * num_systems is not the same with len(defocus_range.)
    # In this case, I have to know how np.split(data, sections) works.
    # Sections [n1, n2, n3, n4]라면, [0: n1], [n1:n2], [n2:n3], [n3:n4], [n4:] 이렇게 가져온다. 균일하게 나눠져 있지 않아도 ㄱㅊ.
    sections = [i * chunk_size for i in range(1, num_systems)]
    defocus_ranges = np.split(defocus_range, sections)
    zidxs = np.arange(len(defocus_range), dtype=np.uint64)
    chunked_zidxs = np.split(zidxs, sections)
    chunk_sizes = [len(zidxs) for zidxs in chunked_zidxs] 
    placeholders = 'placeholder_deconvs' in optdeconv
    
    if placeholders:
        deconvs = optdeconv["placeholder_deconvs"]
    else:
        deconvs = optdeconv["deconvs"]
    scheduler = lr_scheduler is not None
    
    # define logging messages
    log_string = "[{}] iter: {}, loss: {}, norm(phase mask): {}"
    profile_string = "[{}] {}"
    
    # create MSE loss
    mse = nn.MSELoss()
    
    # create high pass filter if specified.
    if high_pass_kernel_size > 0:
        with torch.no_grad():
            ks = high_pass_kernel_size
            delta_kernel = torch.zeros(ks, ks, device='cpu')
            delta_kernel[int(ks / 2), int(ks / 2)] = 1.0
            low_pass_kernel = gaussian_kernel_2d(
                int(ks / 2) + 1, (ks, ks), device='cpu'
            )
            high_pass_kernel = delta_kernel - low_pass_kernel
            high_pass_kernel = high_pass_kernel.expand(
                num_grad_recon_planes, *high_pass_kernel.shape
            ) # 실제 gradient를 구할 수만큼 구성.
    else:
        high_pass_kernel = None
        
    # initialize end time
    end_time = None
    
    # train loop
    while it < num_iterations:
        for data in dataloader:
            start_time = time.perf_counter()
            sam = data[0] # sample
            if end_time is None:
                logging.info(
                    profile_string.format(start_time, "done loading new sample")
                )
            else:
                logging.info(
                    profile_string.format(
                        (start_time - end_time), "done loading new sample"
                    )
                )
            
            optimizer.zero_grad()
            
            # load trainable phase mask
            phase_mask_angle = optdeconv["phase_mask"]()
            # phase mask with mirror
            if "mirror" in optdeconv:
                phase_mask_angle = phase_mask_angle + optdeconv['mirror']()
            
            # select image planes with grad.
            # Partially selected indices for separated chunks.
            grad_im_chunk_idxs = select_chunk_indices(
                num_systems, chunk_sizes, num_grad_im_planes
            )
            grad_im_zs = take_chunk_indices(defocus_ranges, grad_im_chunk_idxs)
            
            # Select data without gradient
            nograd_im_chunk_idxs = chunk_indices_diff(grad_im_chunk_idxs, chunk_sizes)
            nograd_im_zs = take_chunk_indices(defocus_range, nograd_im_chunk_idxs)
            # 하나라도 True면 True : any
            do_nograd_im = any(len(idxs)>0 for idxs in nograd_im_chunk_idxs)
            if do_nograd_im:
                # Actually, Scattering each sample to corresponding device.
                nograd_im_sams = selective_scatter(sam, nograd_im_chunk_idxs, devices)
                #### Paralleization for 4F system simulation
                nograd_ims = parallel_apply(
                    optdeconv['opts'],
                    [
                        (phase_mask_angle, sam, zs) for (sam, zs) in zip(nograd_im_sams, nograd_im_zs)
                    ],
                    devices,
                    requires_grad=False
                )
                
                nograd_ims = [im.unsqueeze(0) for im in nograd_ims]
                nograd_ims = gather(nograd_ims, devices[0]) # devices[0]에 nograd_ims를 모은다.
            ### samples with grad in parallel on multi-gpus
            grad_im_sams = selective_scatter(sam, grad_im_chunk_idxs, devices)
            grad_ims = parallel_apply(
                optdeconv['opts'],
                [
                    (phase_mask_angle, sam, zs)
                    for (sam, zs) in zip(grad_im_sams, grad_im_zs)
                ],
                devices,
                requires_grad=True
            )
            ## Same procedure
            grad_ims = [im.unsqueeze(0) for im in grad_ims]
            grad_ims = gather(grad_ims, devices[0])
            
            
            if do_nograd_im:
                ims = torch.cat([grad_ims, nograd_ims])
            else:
                ims = grad_ims
                
            ### This is because, 3D to 2D compression. We splited the 3D data in depth axis.
            im = torch.sum(ims, axis=0)
            ### TODO: 이어서.