import glob
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import math
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from monai.transforms.transform import MapTransform, Transform
from monai.data import DataLoader, Dataset, CacheDataset
from monai.transforms import (
    AddChanneld, Compose, LoadImaged, Spacingd, Orientationd, ResizeWithPadOrCropd, Resized, ScaleIntensityRanged,
    ToTensord, CastToTyped, ScaleIntensityd, CropForegroundd, RandSpatialCropSamplesd
)
from monai.utils import first, set_determinism
from monai.config import KeysCollection


def load_data_task03(data_dir, **kwargs):
    set_determinism(seed=2021)
    data_files = sorted(glob.glob(os.path.join(data_dir, "aligned_norm.nii.gz"), recursive=True))
    scan_files = []
    seg_files = []
    for scan_file in data_files:
        dir = os.path.dirname(scan_file)
        seg_file = os.path.join(dir, "aligned_seg35.nii.gz")
        if os.path.exists(seg_file):
            scan_files.append(scan_file)
            seg_files.append(seg_file)
    print(len(data_files), len(scan_files), len(scan_files))
    data_dicts = [{"image": scan_file,
                   "label": seg_file}
                  for (scan_file, seg_file) in zip(scan_files, seg_files)]
    data_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),  # reorient to RAS will need post-processing
        ToTensord(keys=["image", "label"]),
        CastToTyped(keys=["label"], dtype=torch.int64),
        CastToTyped(keys=["imaage"], dtype=torch.float32)
    ])
    dataset = CacheDataset(data=data_dicts, transform=data_transforms, **kwargs)
    return dataset, data_transforms


class DistanceTransform(Transform):
    """
    Compute the distance transform of array_like input
    """

    def __init__(self, sampling: Optional[Union[int, float, Tuple[int, ...], List[int]]] = None) -> None:
        """
        Args:
            sampling: Spacing of elements along each dimension. If a sequence, must be of length equal to the input rank;
            if a single number, this is used for all axes. If not specified, a grid spacing of unity is implied.
        """
        self.sampling = sampling

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[Tuple[np.ndarray], Tuple[torch.Tensor]]:
        ndims = len(img.shape)
        if ndims > len(self.sampling):
            img = img.squeeze()
        if isinstance(img, np.ndarray):
            bg = np.where(img == 0, 1, 0)
        elif isinstance(img, torch.Tensor):
            bg = torch.where(img == 0, 1, 0)
        else:
            raise TypeError(f"img must be one of (numpy.ndarray, torch.Tensor) but is {type(img).__name__}.")
        fg_dt = distance_transform_edt(img, self.sampling)
        bg_dt = distance_transform_edt(bg, self.sampling)
        if ndims > len(self.sampling):
            fg_dt = np.expand_dims(fg_dt, 0)
            bg_dt = np.expand_dims(bg_dt, 0)
        if isinstance(img, np.ndarray):
            return fg_dt, bg_dt
        if isinstance(img, torch.Tensor):
            return torch.from_numpy(fg_dt), torch.from_numpy(bg_dt)


class DistanceTransformd(MapTransform):
    """
    The distance transform input specified by 'keys' (segmentation label)
    will be computed and stored.
    """

    def __init__(
            self,
            keys: KeysCollection,
            sampling: Optional[Union[int, float, Tuple[int, ...], List[int]]] = None,
            output_prefixes: Optional[Sequence[str]] = None,
            allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_prefixes: the prefixes to construct keys to store distance transform
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        assert len(keys) == len(output_prefixes)
        self.output_prefixes = output_prefixes
        self.distance_transform = DistanceTransform(sampling)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
                 ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key, prefix in zip(self.key_iterator(d), self.output_prefixes):
            # squeeze channel
            fg_dt, bg_dt = self.distance_transform(d[key])
            d[f"{prefix}_fg_dt"] = fg_dt
            d[f"{prefix}_bg_dt"] = bg_dt
        return d

# if __name__ == "__main__":
    # dataset = load_data_task03(data_dir="/mnt/bailiang/learn2reg/task3/neurite-oasis.v1.0/**/", cache_rate=0.01,
    #                            num_workers=2)
    # train_dataset, val_dataset = dataset[:19], dataset[-20:]
    # print(len(train_dataset))
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # print(len(train_loader))
    # check_data = first(train_loader)
    # dt = check_data["lab_fg_dt"][0][0]
    # shape = dt.shape
    # # plt.imshow(dt[shape[0]//2, :, :])
    # # plt.show()
    # print(dt.max(), dt.min())
    # test = torch.where(dt==0, 0.0, 1-(1-1/dt)**2)
    # print(test.max(), test.min())
    # plt.imshow(test[shape[0]//2, :, :])
    # plt.colorbar()
    # plt.show()
    # for data in train_loader:
    #     data1 = data["image"][0][0]
    #     data2 = data["image"][1][0]
    #     print((data1-data2).mean())
    #     shape = data1.shape
    #     plt.figure("check", (12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(data1[shape[0] // 2, :, :], cmap="gray")
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(data2[shape[0] // 2, :, :], cmap="gray")
    #     plt.show()
