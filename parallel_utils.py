import threading

import numpy as np
import torch
from torch._utils import ExceptionWrapper
import torch.cuda.comm
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda._utils import _get_device_index
from utils import _single

def gather(outputs, device):
    return nn.parallel.gather(outputs, _get_device_index(device))

# TODO: interpretation
def parallel_apply(modules, inputs, devices, requires_grad=True):
    lock = threading.Lock()
    results = {}
    
    ### Each parallel playing. - save the result in results dict.
    def _worker(i, module, scattered_input, device, requires_grad):
        try:
            if device.type != 'cpu':
                torch.backends.cuda.cufft_plan_cache[i].clear()
            if isinstance(module, nn.Module):
                module.to(device)
            scattered_input = [
                s.to(device) if isinstance(s, torch.Tensor) else s
                for s in scattered_input
            ]
            if requires_grad:
                result = module(*scattered_input)
            else:
                with torch.no_grad():
                    result = module(*scattered_input)
            with lock:
                results[i] = result
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where=f"in module {i} on device {device}")

    if len(modules) > 1:
        threads = [
            threading.Thread(
                target=_worker,
                args=(
                    i,
                    module,
                    _single(scattered_input),
                    device,
                    requires_grad,
                ),
            )
            for i, (module, scattered_input, device) in enumerate(
                zip(modules, inputs, devices)
            )
        ]
        # start thread...
        for thread in threads:
            thread.start()
        # join thread...
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], _single(inputs[0]), devices[0], requires_grad)
    
    # during the above process, in results dict, results were saved.
    outputs = []
    for i in range(len(modules)):
        result = results[i]
        if isinstance(result, ExceptionWrapper):
            result.reraise()
        outputs.append(result)
    return outputs
        
        
    
def select_chunk_indices(num_chunks, chunk_sizes, num_per_chunk=None):
    if num_per_chunk is None:
        return [np.arange(chunk_size) for chunk_size in chunk_sizes] # chunk_sizes: [N, N, N, N,... n]
    else:
        #### -> sparse selection을 위함.
        return [
            np.sort(
                np.random.choice(np.arange(chunk_size), num_per_chunk, replace=False)
            )
            for chunk_size in chunk_sizes
        ]

def chunk_indices_diff(chunk_idxs, chunk_sizes):
    # Returns the chunk indices that were not chosen in select_chunk_indices for chunks for size chunk_size
    # return [
    #     np.setdiff1d(idxs, np.arange(chunk_size), assume_unique=True)
    #     for idxs, chunk_size in zip(chunk_idxs, chunk_sizes)
    # ]
    # 아래 코드는 chunk_sizes의 범위 내에서, idxs가 아닌것들을 골라내는 것임.
    # chunk_idxs가 이전에 골랐던 indices들이기 때문에, 아래로 코딩하는게 맞음.
    return [
        np.setdiff1d(np.arange(chunk_size), idxs, assume_unique=True)
        for idxs, chunk_size in zip(chunk_idxs, chunk_sizes)
    ]

def take_chunk_indices(split_array, chunk_idxs):
    ### Each chunk data: Split array [N * part_array]
    ### Selected indices in each chunk: chunk_idxs [N * partpart_indices_num]
    return [array[idxs] for array, idxs in zip(split_array, chunk_idxs)]

def selective_scatter(array, chunk_idxs, devices):
    """
    Scattering the array to each device
    """
    chunk_size = int(len(array) / len(devices))
    if len(array) % len(devices) != 0:
        chunk_size += 1
    split_array = torch.split(array, chunk_size)
    return [
        array[idxs].to(d) for array, idxs, d in zip(split_array, chunk_idxs, devices)
    ]