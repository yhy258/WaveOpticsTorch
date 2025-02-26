import random
import numpy as np
import torch


### TODO: Augment 기법들  공부 필요.
def cylinder_crop(
    vol, radius, center=None, max_depth=101, valid_ratio=0.75, probability=1.0
):
    if (probability < 1.0) and (random.random() >= probability):
        return vol
    vol = torch.tensor(vol)
    if center == None:
        center = [
            random.randint(
                int((1 - valid_ratio) / 2 * vol.shape[d]),
                int((1 + valid_ratio) / 2 * vol.shape[d]),
            )
            for d in [1, 2]
        ]
    elif center == "center":
        center = [int(d / 2) for d in vol.shape[1:]]
    vol = vol[: min(max_depth, vol.shape[0])]
    y, x = torch.meshgrid(
        [
            torch.arange(s, dtype=torch.float32, device=vol.device)
            for s in (2 * radius + 1, 2 * radius + 1)
        ]
    )
    y = y - radius
    x = x - radius
    dist = (y.pow(2) + x.pow(2)).sqrt()
    circle = (dist <= radius).unsqueeze(0).to(torch.float32)
    shape = [
        min(center[-2] + radius + 1, vol.shape[1]) - max(center[-2] - radius, 0),
        min(center[-1] + radius + 1, vol.shape[2]) - max(center[-1] - radius, 0),
    ]
    cylinder = torch.zeros(vol.shape[0], shape[0], shape[1])
    circle = circle[:, : shape[0], : shape[1]]
    mask = circle.expand(*cylinder.shape)
    cylinder[:, :, :] = vol[
        :,
        max(center[-2] - radius, 0) : min(center[-2] + radius + 1, vol.shape[1]),
        max(center[-1] - radius, 0) : min(center[-1] + radius + 1, vol.shape[2]),
    ]
    cylinder *= mask
    return cylinder.squeeze().numpy()


def pad_volume(vol, target_shape, mode="constant", estimate=False, **kwargs):
    background = 0
    if estimate:
        background = np.mean(vol[0:10, 0:10, 0:10])
    return np.pad(
        vol,
        [
            (int((t - s) / 2), int((t - s + 1) / 2))
            for s, t in zip(vol.shape, target_shape)
        ],
        mode=mode,
        constant_values=background,
        **kwargs,
    )
