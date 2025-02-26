import os
from glob import glob
from itertools import cycle, islice
import time
import skimage
import torch
import zarr
from torch.utils.data import Dataset

import random

import numpy as np
from itertools import repeat

### UTILS
def _list(x, repetitions=1):
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return x
    else:
        return [
            x,
        ] * repetitions

def _ntuple(n):
    """Creates a function enforcing ``x`` to be a tuple of ``n`` elements."""

    def parse(x):
        if isinstance(x, tuple):
            return x
        return tuple(repeat(x, n))

    return parse

_triple = _ntuple(3)


class ROI:
    """
    ROI (region of interest) implementation based on ``daisy.Roi`` from
    https://github.com/funkelab/daisy.
    """

    def __init__(self, extents, pixel_size=None, pixel_offset=None):
        self.extents = extents
        if pixel_size is None:
            pixel_size = (1.0,) * len(self.extents)
        if pixel_offset is None:
            pixel_offset = (0,) * len(self.extents)
        self.pixel_size = pixel_size
        self.pixel_offset = pixel_offset

    def __repr__(self):
        return "\n".join(
            [
                f"ROI ({self.size} @ {self.pixel_size})",
                f"extents: {self.extents}",
                f"shape: {self.shape}",
                f"pixel_offset: {self.pixel_offset}",
                "=====================================",
            ]
        )

    @property
    def size(self):
        return tuple(end - start for (start, end) in self.extents)

    @property
    def shape(self):
        return tuple(int((sz) / self.pixel_size[i]) for i, sz in enumerate(self.size))

    def dims(self):
        return len(self.shape)

    def start(self):
        return tuple(s if s is not None else -np.inf for (s, e) in self.extents)

    def end(self):
        return tuple(e if e is not None else np.inf for (s, e) in self.extents)

    def volume(self):
        return np.prod(self.size)

    def empty(self):
        return all(tuple((end - start) == 0 for (start, end) in self.extents))

    def contains(self, other):
        if not isinstance(other, ROI):
            return all(
                [other[d] >= self.start()[d] for d in range(self.dims())]
            ) and all([other[d] < self.end()[d] for d in range(self.dims())])
        if other.empty():
            return self.contains(other.start())
        else:
            return self.contains(other.start()) and self.contains(
                other.end() - (1,) * other.dims()
            )

    def intersects(self, other):
        assert self.dims() == other.dims(), "ROIs must have same dims"
        if self.empty() or other.empty():
            return False
        separated = any(
            [
                ((None not in [s1, s2, e1, e2]) and ((s1 >= e2) or (s2 >= e2)))
                for (s1, s2, e1, e2) in zip(
                    self.start(), other.start(), self.end(), other.end()
                )
            ]
        )
        return not separated

    def intersect(self, other):
        if not self.intersects(other):
            return ROI([(0, 0) for d in self.dims()], self.pixel_size)
        start = tuple(max(s1, s2) for s1, s2 in zip(self.start(), other.start()))
        end = tuple(min(e1, e2) for e1, e2 in zip(self.end(), other.end()))
        start = tuple(s if s > -np.inf else None for s in start)
        end = tuple(e if e < np.inf else None for e in end)
        extents = [(s, e) for (s, e) in zip(start, end)]
        pixel_offset = tuple(
            (self.pixel_offset[i] + int(max(0, s2 - s1) / self.pixel_size[i]))
            for (i, (s1, s2)) in enumerate(zip(self.start(), start))
        )
        return ROI(extents, pixel_size=self.pixel_size, pixel_offset=pixel_offset)

    def union(self, other):
        start = tuple(min(s1, s2) for s1, s2 in zip(self.start(), other.start()))
        end = tuple(max(e1, e2) for e1, e2 in zip(self.end(), other.end()))
        start = tuple(s if s > -np.inf else None for s in start)
        end = tuple(e if e < np.inf else None for e in end)
        extents = [(s, e) for (s, e) in zip(start, end)]
        pixel_offset = tuple(
            min(p1, p2) for p1, p2 in zip(self.pixel_offset, other.pixel_offset)
        )
        return ROI(extents, pixel_size=self.pixel_size, pixel_offset=pixel_offset)

    def shift(self, by):
        extents = [(s + b, e + b) for (b, (s, e)) in zip(by, self.extents)]
        return ROI(extents, self.pixel_size)

    def grow(self, amount_neg, amount_pos):
        if amount_neg is None:
            amount_neg = (0,) * self.dims()
        if amount_pos is None:
            amount_pos = (0,) * self.dims()
        amount_neg = _list(amount_neg, repetitions=self.dims())
        amount_pos = _list(amount_pos, repetitions=self.dims())
        pixel_offset = tuple(
            self.pixel_offset[i] + int(n / self.pixel_size[i])
            for i, n in enumerate(amount_neg)
        )
        start = tuple(s + amount_neg[i] for i, s in enumerate(self.start()))
        end = tuple(e + amount_pos[i] for i, e in enumerate(self.end()))
        start = tuple(s if s > -np.inf else None for s in start)
        end = tuple(e if e < np.inf else None for e in end)
        extents = [(s, e) for (s, e) in zip(start, end)]
        return ROI(extents, pixel_size=self.pixel_size, pixel_offset=pixel_offset)

    def interpolate(self, pixel_size):
        """
        Creates new ROI with same extent in real units but different shape
        resulting from the specified ``pixel_size``.
        """
        return ROI(self.extents, pixel_size=pixel_size)

    def random_location(self):
        return tuple(random.uniform(s, e) for (s, e) in self.extents)

    def center_location(self):
        return tuple(0.5 * (s + e) for (s, e) in self.extents)

    def to_pixels(self, location):
        pixel_offset = self.pixel_offset
        pixel_size = self.pixel_size
        return tuple(
            (pixel_offset[i] + int((loc - s - pixel_size[i] * 1e-6) / pixel_size[i]))
            for i, (loc, (s, e)) in enumerate(zip(location, self.extents))
        )



class ConfocalVolumesDataset(Dataset):
    def __init__(
        self,
        paths: list,
        shape: tuple,
        steps: tuple = (1, 1, 1),
        location: str = "random",
        strategy: str = "center",
        bounds_roi: str = "data",
        valid_ratio: float = 1.0,
        augmentations=None,
        exclude=None,
        balance: bool = True,
        repeat: int = 0,
        cache: bool = True,
        num_tries: int = 2,
        return_dict: bool = False, 
    ):
        super().__init__()
        
        self.paths = [os.path.realpath(path) for path in paths]
        self.fnames = [
            sorted(list(glob(os.path.join(path, "*.zarr")))) for path in self.paths
        ]
        if balance:
            max_volumes = max([len(flist) for flist in self.fnames])
            self.fnames = [
                list(islice(cycle(flist), max_volumes)) for flist in self.fnames
            ] # 모든 파일의 수를 맞추기.
            
        self.fnames = [fname for flist in self.fnames for fname in flist]
        if exclude is not None:
            self.fnames = [f for f in self.fnames if f not in exclude] # 뺄 filenames 제외.
        
        self.repeat = repeat
        if self.repeat > 0:
            self.fnames = self.fnames * repeat
        self.shape = shape
        self.steps = steps
        assert location in [
            "random",
            "center",
        ], "location must be either 'random' or 'center'"
        self.location = location
        assert bounds_roi in ["data"] or isinstance(
            bounds_roi, ROI
        ), "bounds_roi must be either 'data' or an ROI instance"
        self.strategy = strategy
        assert strategy in [
            "center",
            "top",
        ], "strategy must be either 'center' or 'top'"
        self.read_function = eval(f"self.from_{self.strategy}")
        self.bounds_roi = bounds_roi
        self.valid_ratio = _triple(valid_ratio)
        if augmentations is not None:
            augmentations = _list(augmentations)
            
        self.augmentations = augmentations
        self.cache = cache
        self.cached_data = [None for i in range(len(self.fnames))]
        assert num_tries > 1, "Number of tries must be >= 1"
        self.num_tries = num_tries
        self.return_dict = return_dict
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        for _ in range(self.num_tries):
            try:
                if self.cache and self.cached_data[index] is not None:
                    vol_group, vol_roi = self.cached_data[index]
                else:
                    vol_group, vol_roi = self.read_volume(self.fnames[index])
                if self.cache:
                    vol_group = vol_group[...]
                    self.cached_data[index] = (vol_group, vol_roi)
                bounds_roi = self.bounds_roi
                if bounds_roi == "data":
                    bounds_roi = self.safe_bounds(vol_roi)
                vol = self.read_function(vol_group, vol_roi, bounds_roi, self.location)
                # apply augmentations
                if self.augmentations is not None:
                    augment = self.augmentations[index % len(self.augmentations)]
                    out = augment(vol)
                    if isinstance(out, tuple):
                        vol = out[0]
                        logs = out[1]
                        logs.append(self.fnames[index])
                    if self.return_dict:
                        return {
                            "data": torch.from_numpy(
                                out, dtype=torch.float32
                            ).unsqueeze(0)
                        }
                    else:
                        return out
                else:
                    if self.return_dict:
                        return {
                            "data": torch.from_numpy(
                                vol, dtype=torch.float32
                            ).unsqueeze(0)
                        }
                    else:
                        return vol
            except:
                time.sleep(0.5)
                pass
        ### Raise 하게끔 수정함. -> 나중에 자리나면 ㄱㄱ.!
        raise Exception(f"DATA LOAD FAILURE: INDEX: {index}, FILE: {self.fnames[index]}")
                    
                    
    def read_volume(self, fname):
        zroot = zarr.open(fname, mode='r')
        vol_group = zroot["volume/data"]
        if "volume/attrs" in zroot:
            attrs_group = zroot["volume/attrs"]
            attrs = attrs_group.attrs
            vol_roi = ROI(attrs["extents"], attrs["pixel_size"])
        else:
            vol_roi = ROI([(0,d) for d in vol_group.shape])
        return vol_group, vol_roi
    
    def safe_bounds(self, vol_roi):
        sizes = (0,)*3
        if self.shape is not None:
            sizes = tuple(sh * vol_roi.pixel_size[i] for i, sh in enumerate(self.shape))
        bounds_roi = vol_roi.grow([sz / 2 for sz in sizes], [-sz / 2 for sz in sizes])
        bounds_roi = bounds_roi.grow(
            [(1 - r) * 0.5 * sz for r, sz in zip(self.valid_ratio, bounds_roi.size)],
            [-(1 - r) * 0.5 * sz for r, sz in zip(self.valid_ratio, bounds_roi.size)],
        )
        bounds_roi = bounds_roi.intersect(vol_roi)
        return bounds_roi
    
    def from_center(self, vol_group, vol_roi, bounds_roi, location_func="random"):
        if not bounds_roi.empty() and self.shape is not None:
            shape = self.shape
            steps = tuple(int(s) for s in self.steps)
            location_command = f"(bounds_roi.{location_func}_location())"
            center = bounds_roi.to_pixels(eval(location_command))
            slices = tuple(
                slice(
                    center[d] - int(shape[d] / 2),
                    center[d] + int(shape[d] / 2),
                    steps[d],
                )
                for d in range(3)
            )
            clipped_slices = []
            for d in range(3):
                start = max(0, center[d] - int(shape[d] / 2))
                end = min(center[d] + int(shape[d] / 2), vol_roi.shape[d])
                clipped_slices.append(slice(start, end, steps[d]))
            clipped_slices = tuple(clipped_slices)
            vol = vol_group[clipped_slices]
            pads = []
            pad = False
            for d in range(3):
                to_pad = [
                    round((clipped_slices[d].start - slices[d].start) / steps[d]),
                    round((slices[d].stop - clipped_slices[d].stop) / steps[d]),
                ]
                if any([p > 0 for p in to_pad]):
                    pad = True
                    if sum(to_pad) != ((self.shape[d] / steps[d]) - vol.shape[d]):
                        to_pad[-1] += 1
                pads.append(tuple(to_pad))
            if pad:
                background = np.mean(vol_group[0:10, 0:10, 0:10])
                vol = skimage.util.pad(vol, pads, constant_values=background)
            vol = vol[0 : int(shape[0] / steps[0])]
            return vol
        elif bounds_roi.empty():
            shape = vol_roi.shape
            steps = tuple(int(s) for s in self.steps)
            slices = tuple(slice(0, shape[d], steps[d]) for d in range(3))
            vol = vol_group[slices]
            return vol
        else:
            return vol_group[:]

    def from_top(self, vol_group, vol_roi, bounds_roi, location_func="random"):
        if not bounds_roi.empty() and self.shape is not None:
            shape = self.shape
            steps = tuple(int(s) for s in self.steps)
            location_command = f"(bounds_roi.{location_func}_location())"
            center = bounds_roi.to_pixels(eval(location_command))
            slices = tuple(
                slice(
                    center[d] - int(shape[d] / 2),
                    center[d] + int(shape[d] / 2),
                    steps[d],
                )
                for d in [1, 2]
            )
            slices = (slice(0, self.shape[0], steps[0]), *slices)
            clipped_slices = [slice(0, min(self.shape[0], vol_roi.shape[0]), steps[0])]
            for d in [1, 2]:
                start = max(0, center[d] - int(shape[d] / 2))
                end = min(center[d] + int(shape[d] / 2), vol_roi.shape[d])
                clipped_slices.append(slice(start, end, steps[d]))
            clipped_slices = tuple(clipped_slices)
            vol = vol_group[clipped_slices]
            pads = []
            pad = False
            for d in range(3):
                to_pad = [
                    round((clipped_slices[d].start - slices[d].start) / steps[d]),
                    round((slices[d].stop - clipped_slices[d].stop) / steps[d]),
                ]
                if any([p > 0 for p in to_pad]):
                    pad = True
                    if sum(to_pad) != ((self.shape[d] / steps[d]) - vol.shape[d]):
                        to_pad[-1] += 1
                pads.append(tuple(to_pad))
            if pad:
                background = np.mean(vol_group[0:10, 0:10, 0:10])
                vol = skimage.util.pad(vol, pads, constant_values=background)
            return vol
        elif bounds_roi.empty():
            shape = vol_roi.shape
            steps = tuple(int(s) for s in self.steps)
            slices = tuple(slice(0, shape[d], steps[d]) for d in range(3))
            vol = vol_group[slices]
            return vol
        else:
            return vol_group[:]

#%%
# import os
# import numpy as np
# import zarr

# root = '/data/joon/Dataset/FourierNetData/LarvalJebrafish/interpolated_1.625um/zjabc_train'
# sub_directories = os.listdir(root)
# file_paths = [os.path.join(root, sub_dir, sub_file) for sub_dir in sub_directories for sub_file in os.listdir(os.path.join(root, sub_dir))]
# datas = []
# for i, d in enumerate(file_paths):
#     data = zarr.open(file_paths[i])
#     datas.append(data["volume/data"][:])

