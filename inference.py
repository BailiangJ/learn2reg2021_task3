import os
import torch
import numpy as np
import gc
import time
import ml_collections

os.environ["VXM_BACKEND"] = 'pytorch'
from data_utils import load_data_task03
from deepregnet import RegNet
from monai.data import DataLoader
from monai.metrics import compute_meandice
from monai.networks import one_hot
from pathlib import Path
from pandas import DataFrame, read_csv
from voxelmorph.torch import layers
import nibabel as nib
from evaluation.surface_distance import compute_dice_coefficient
from scipy.ndimage.interpolation import zoom, map_coordinates
import SimpleITK as sitk
import matplotlib.pyplot as plt

device = 'cpu'
exp_id = 66308
data_dir = "/mnt/bailiang/learn2reg/task3/neurite-oasis.v1.0/**/"
model_path = f"output/{exp_id}/saved_models/0050.pt"
output_dir = Path(f"output/{exp_id}/inference/")
pairs_path = "/mnt/bailiang/learn2reg/task3/pairs_val.csv"

pairs = DataFrame(read_csv(pairs_path, skipinitialspace=True, encoding="utf-8").to_dict(orient="records"))

dataset = load_data_task03(data_dir, cache_rate=0.01, num_workers=2)
val_dataset1 = dataset[-20:-1]
val_dataset2 = dataset[-19:]
val_loader1 = DataLoader(val_dataset1, batch_size=1, shuffle=False)
val_loader2 = DataLoader(val_dataset2, batch_size=1, shuffle=False)
#
model = RegNet.load(model_path, device)

model.eval()
with torch.no_grad():
    for (_, row), fixed_data, moving_data in zip(pairs.iterrows(), val_loader1, val_loader2):
        start_time = time.time()
        source = moving_data["image"].to(device).float()
        target = fixed_data["image"].to(device).float()
        source_mask = moving_data["label"].to(device).float()
        target_mask = fixed_data["label"].to(device).float()

        y_source_mask, flow = model.inference(source, target, source_mask)

        target_mask_oh = one_hot(target_mask, num_classes=36)
        y_source_mask_oh = one_hot(y_source_mask, num_classes=36)
        dice_score = compute_meandice(target_mask_oh, y_source_mask_oh, include_background=False)
        print(f"dice_score:{dice_score.mean().item()}")

        # orientation from "RAS" to "LIA"
        flow = flow.squeeze()
        flow = torch.flip(flow, [1, 3])
        flow[0] = -flow[0]
        flow[2] = -flow[2]
        flow = flow.permute(0, 1, 3, 2)
        flow = flow[[0, 2, 1]]

        flow = flow.numpy().astype(np.float16)
        fname = output_dir / f"disp_{row['fixed']:04d}_{row['moving']:04d}.npz"
        np.savez(fname, flow)
        print(f"time: {time.time() - start_time:0.4f} sec")

