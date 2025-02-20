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
from system4f import compute_power


def train_psf(
    optdeconv,
    optimizer,
    dataloader,
    defocus_range,
    num_grad_im_planes, # these are for sparse-gradient calculation.
    num_grad_recon_planes, # these are for sparse-gradient calculation.
    devices,
    losses,
    high_pass_losses,
    regularized_losses,
    num_iterations,
    lr_scheduler=None,
    single_decoder=False,
    input_3d=False,
    im_function=None, # any additional imaging function
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
            
            # add shot noise to image
            im = poissonlike_gaussian_noise(im)
            if im_function is not None:
                im = im_function(im) # e.g. masking img, normalization, ...
            
            if input_3d:
                im = im.view(1, 1, 1, *im.shape) # 1, 1, 1, H, W
            else:
                im = im.view(1, 1, *im.shape)
                
            image_time = time.perf_counter()
            logging.info(
                profile_string.format((image_time - start_time), "done imaging sample")
            )

            # select gradient reconstruction planes
            recon_chunk_idxs = select_chunk_indices(
                num_systems, chunk_sizes, num_grad_recon_planes # 실제로 골라진 recon planes.
            )
            recon_chunk_zidxs = take_chunk_indices(chunked_zidxs, recon_chunk_idxs)
            selected_sams = selective_scatter(sam, recon_chunk_idxs, devices)
            
            if low_pass_weight > 0 or high_pass_kernel is None:
                # calculate normalization factor only
                normalization = torch.mean(gather(selected_sams, devices[0]) ** 2)
            if high_pass_kernel is not None:
                # high pass filter sams and then calculate normalization factor
                high_pass_selected_sams = parallel_apply(
                    [high_pass_filter for d in devices],
                    [(sam, high_pass_kernel) for sam in selected_sams],
                    devices,
                )
                high_pass_normalization = torch.mean(
                    gather(high_pass_selected_sams, devices[0]) ** 2
                )
            else:
                # placeholder arguments for parallel loss calculation
                # if we don't need high passed samples
                high_pass_selected_sams = [None for sam in selected_sams]
            
            if placeholders:
                copy_deconv_params_to_placeholder(num_systems, optdeconv, recon_chunk_zidxs)
                
            scattered_recons, scattered_losses = zip(
                *parallel_apply(
                    [chunk_recon_and_loss for device in devices],
                    [
                        (
                            deconv,
                            im,
                            sam,
                            high_pass_sam,
                            mse,
                            low_pass_weight,
                            high_pass_kernel,
                            single_decoder,
                            input_3d
                        )
                        for (deconv, sam, high_pass_sam) in zip(
                            deconvs, selected_sams, high_pass_selected_sams
                        )
                    ]
                )
            )
            
            recon_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (recon_time - image_time),
                    "done reconstructing sample and performing distributed loss backwards",
                )
            )
            
            
            # normalize loss
            if high_pass_kernel is not None:
                scattered_high_pass_loss = [
                    l["high_pass_loss"].to(devices[0]) for l in scattered_losses
                ]
                high_pass_loss = torch.cuda.comm.reduce_add(
                    scattered_high_pass_loss, destination=_get_device_index(devices[0])
                ) / len(scattered_high_pass_loss)
                high_pass_loss = high_pass_loss / high_pass_normalization
            if low_pass_weight > 0 or high_pass_kernel is None:
                scattered_loss = [l["loss"].to(devices[0]) for l in scattered_losses]
                loss = torch.cuda.comm.reduce_add(
                    scattered_loss, destination=_get_device_index(devices[0])
                ) / len(scattered_loss)
                loss = loss / normalization
            if high_pass_kernel is not None:
                unregularized_loss = high_pass_loss
            else:
                unregularized_loss = loss
            if low_pass_weight > 0:
                unregularized_loss = unregularized_loss + (low_pass_weight * loss)
            
            if regularize_lost_power > 0:
                lost_powers = parallel_apply(
                    [chunk_lost_power for device in devices],
                    [(opt.psf, opt.cropoped_psf, opt.dk) for opt in optdeconv["opts"]], devices,
                )
                lost_power = torch.sum(
                    torch.stack([power.to(devices[0]) for power in lost_powers])
                )
                regularized_loss = unregularized_loss + regularize_lost_power * lost_power
        else:
            regularize_lost_power = unregularized_loss
        loss_time = time.perf_counter()
        logging.info(
                profile_string.format((loss_time - recon_time), "done averaging loss")
        )
        regularized_loss.backward()
        backward_time = time.perf_counter()
        logging.info(
                profile_string.format((backward_time - loss_time), "done with backward")
            )
        # update weights
        optimizer.step()
        
        if scheduler:
            # advance learning rate schedule
            lr_scheduler.step()
        if placeholders:
            # copy params back from placeholders on multiple gpus
            copy_placeholder_params_to_deconv(
                num_systems, optdeconv, recon_chunk_zidxs
            )
        opt_time = time.perf_counter()
        logging.info(
            profile_string.format(
                (opt_time - backward_time), "done with optimizer step"
            )
        )
        ### Checkpoint, overall save
        pmnorm = phase_mask_angle.pow(2).sum().detach().cpu().item()
        regularized_losses.append(regularized_loss.detach().cpu().item())
        if high_pass_kernel is not None:
            high_pass_losses.append(high_pass_loss.detach().cpu().item())
        if low_pass_weight > 0 or high_pass_kernel is None:
            losses.append(loss.detach().cpu().item())
        logging.info(
            log_string.format(
                datetime.datetime.now(), it, regularized_losses[-1], pmnorm
            )
        )
        # checkpoint
        if it % 10000 == 0 and it != 0 and validate_args is not None:
            per_sample_val_losses = test_recon(optdeconv, **validate_args) # TODO:
            validate_losses.append(per_sample_val_losses)
        if it % 500 == 0 and it != 0:
            logging.info("checkpointing...")
            checkpoint(
                optdeconv,
                optimizer,
                {
                    "mses": losses,
                    "high_pass_mses": high_pass_losses,
                    "regularized_losses": regularized_losses,
                    "validate_high_pass_mses": validate_losses,
                },
                it,
            )
        if (it % 10000 == 0) or (it == num_iterations - 1):
            logging.info("snapshotting...")
            with torch.no_grad():
                opt = optdeconv["opts"][0]
                phase_mask = ctensor_from_phase_angle(phase_mask_angle)
                phase_mask = phase_mask.to(devices[0])
                psf = torch.stack(
                    [
                        opt.compute_psf(phase_mask, z).detach().cpu()
                        for z in defocus_range
                    ]
                )
                if not single_decoder:
                    full_recon = torch.stack(
                        [
                            optdeconv["deconvs"][z](im.cpu()).detach().cpu()
                            for (zidxs, d) in zip(chunked_zidxs, devices)
                            for z in zidxs
                        ]
                    )
                else:
                    full_recon = torch.cuda.comm.gather(
                        scattered_recons, destination=-1
                    )
                phase_mask_angle_snapshot = optdeconv["phase_mask"]()
                if "mirror" in optdeconv:
                    mirror_phase_angle_snapshot = optdeconv["mirror"]()
                else:
                    mirror_phase_angle_snapshot = None
            snapshot(
                optdeconv,
                optimizer,
                {
                    "mses": losses,
                    "high_pass_mses": high_pass_losses,
                    "regularized_losses": regularized_losses,
                    "validate_high_pass_mses": validate_losses,
                },
                it,
                sam,
                full_recon,
                im,
                psf=psf,
                phase_mask_angle=phase_mask_angle_snapshot,
                mirror_phase=mirror_phase_angle_snapshot,
            )

        # update iteration count and check for end
        it += 1
        end_time = time.perf_counter()
        logging.info(
            profile_string.format((end_time - start_time), "done with loop")
        )
        if it >= num_iterations:
            break
        
        

def chunk_recon_and_loss(
    deconvs,
    im,
    sam=None,
    high_pass_sam=None,
    loss_function=F.mse_loss,
    low_pass_weight=0,
    high_pass_kernel=None,
    single_decoder=False, # plane 각각에 대해 복원할건지, 3D 한번에 내놓을건지.
    input_3d=False
):
    if single_decoder:
        recon = deconvs(im)[0]
        recon = recon.squeeze()
        if len(recon.shape)< 3:
            recon = recon.unsqueeze(0)
    else:
        recon_planes = []
        for deconv in deconvs:
            plane = deconv(im)[0]
            plane = plane.squeeze()
            if len(plane.shape) < 3 :
                plane = plane.unsqueeze(0)
            recon_planes.append(plane)
        recon = torch.cat(recon_planes)
    chunk_losses = {}
    if high_pass_kernel is not None:
        high_pass_recon = high_pass_filter(recon, high_pass_kernel)
        # assume high_pass_sam is not None here
        chunk_losses["high_pass_loss"] = loss_function(high_pass_sam, high_pass_recon)
    if low_pass_weight > 0 or high_pass_kernel is None:
        chunk_losses["loss"] = loss_function(sam, recon)
    return (recon, chunk_losses)

def chunk_lost_power(psf, cropped_psf, dk):
    whole_power = compute_power(psf, dk)
    cropped_power = compute_power(cropped_psf, dk)
    lost_power = (whole_power - cropped_power).sum()
    return lost_power

def checkpoint(
    module_container,
    optimizer,
    loss_dict,
    it,
    fname="latest.pt",
    module_name="micdeconv",
):
    """
    Saves the current model/optimizer state.
    """
    checkpoint_dict = {
        f"{module_name}_state_dict": module_container.state_dict(),
        "opt_state_dict": optimizer.state_dict(),
        "it": it,
    }
    checkpoint_dict.update(loss_dict)
    torch.save(checkpoint_dict, fname)

def snapshot(
    module_container,
    optimizer,
    loss_dict,
    it,
    sam,
    recon,
    im,
    psf=None,
    phase_mask_angle=None,
    mirror_phase=None,
    save_dir='snapshots',
    module_name='optdeconv',
):
    checkpoint(
        module_container,
        optimizer,
        loss_dict,
        it,
        fname=f"{save_dir}/state{it}.pt",
        module_name=module_name,
    )
    torch.save(sam.squeeze(), f"{save_dir}/sam{it}.pt")
    torch.save(recon.squeeze(), f"{save_dir}/recon{it}.pt")
    if im is not None:
        torch.save(im.squeeze(), f"{save_dir}/im{it}.pt")
    if psf is not None:
        torch.save(psf.detach().cpu(), f"{save_dir}/psf{it}.pt")
    if phase_mask_angle is not None:
        torch.save(phase_mask_angle.detach().cpu(), f"{save_dir}/phase_mask{it}.pt")
    if mirror_phase is not None:
        torch.save(mirror_phase.detach().cpu(), f"{save_dir}/mirror_phase{it}.pt")
    torch.cuda.empty_cache()
