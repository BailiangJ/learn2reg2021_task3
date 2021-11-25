from typing import Dict, List, Tuple, Optional, Union, Callable
import torch.nn as nn
from torch.optim import Optimizer
from monai.data import DataLoader
from torch.nn.modules.loss import _Loss
from log_utils import LogWriter
import time
import numpy as np
import torch
from monai.networks.utils import one_hot
from monai.metrics import compute_meandice


def run_epoch(model: nn.Module,
              data_loader: DataLoader,
              optimizer: Optimizer,
              loss_func: List[Callable],
              loss_weights: List[float],
              metric_func: List[Callable],
              bidir: bool,
              flipping: bool,
              logWriter: LogWriter,
              device: str,
              phase: str,
              epoch_id: int):
    ncc_loss_list = []
    # Careful!
    mind_loss_list = []
    dice_loss_list = []
    grad_loss_list = []
    dtmse_loss_list = []
    ngf_loss_list = []
    epoch_loss_list = []
    epoch_step_time = []
    sd_logjac_list = []
    metric_func[1].reset()
    metric_func[2].reset()
    metric_func[3].reset()
    for data in data_loader:
        step_start_time = time.time()
        # print(step_start_time)

        source = data["image"][0].unsqueeze(0).float().to(device)
        source_mask = data["label"][0].unsqueeze(0).float().to(device)
        source_dt = data["lab_fg_dt"][0].unsqueeze(0).to(device).float()
        target = data["image"][1].unsqueeze(0).to(device).float()
        target_mask = data["label"][1].unsqueeze(0).to(device).float()
        target_dt = data["lab_fg_dt"][1].unsqueeze(0).to(device).float()

        if flipping and phase == 'train':
            # n_dims = 3
            # is_flip = np.random.randn(n_dims) > 0
            # is_flip = np.array([False, False, *is_flip])
            # axes = np.arange(n_dims + 2)
            #
            # flip_axes = axes[is_flip].tolist()
            # source = torch.flip(source, flip_axes)
            # source_mask = torch.flip(source_mask, flip_axes)
            # source_dt = torch.flip(source_dt, flip_axes)
            # target = torch.flip(target, flip_axes)
            # target_mask = torch.flip(target_mask, flip_axes)
            # target_dt = torch.flip(target_dt, flip_axes)
            if np.random.randn(1) > 0:
                source = torch.flip(source, [2])
                source_mask = torch.flip(source_mask, [2])
                source_dt = torch.flip(source_dt, [2])
                target = torch.flip(target, [2])
                target_mask = torch.flip(target_mask, [2])
                target_dt = torch.flip(target_dt, [2])

        target_mask = one_hot(target_mask, num_classes=36)
        source_mask = one_hot(source_mask, num_classes=36)

        if phase == "train":
            # preint_flow
            if bidir:
                y_source, y_source_mask, \
                y_target, y_target_mask, \
                flow, disp_field = model(source, target,
                                         source_mask, target_mask,
                                         registration=False)
            else:
                y_source, y_source_mask, \
                flow, disp_field = model(source, target,
                                         source_mask, target_mask,
                                         registration=False)
        elif phase == "val":
            # pos_flow
            y_source, y_source_mask, flow = model(source, target,
                                                  source_mask, target_mask,
                                                  registration=True)
            disp_field = flow
        else:
            raise ValueError(f'Unsupported phase: {phase}, available options are ["train", "val"].')

        # device1 = "cuda:1"
        # source = source.to(device1)
        # source_mask = source_mask.to(device1)
        # source_dt = source_dt.to(device1)
        # y_source = y_source.to(device1)
        # y_source_mask = y_source_mask.to(device1)
        # trans_source_dt = trans_source_dt.to(device1)
        # target = target.to(device1)
        # target_mask = target_mask.to(device1)
        # target_dt = target_dt.to(device1)
        # flow = flow.to(device1)

        # if phase == "train" and bidir:
        #     y_target = y_target.to(device1)
        #     y_target_mask = y_target_mask.to(device1)
        #     trans_target_dt = trans_target_dt.to(device1)
        #     ncc_loss = 0.5 * loss_func[0](y_source, target) + 0.5 * loss_func[0](y_target, source)
        #     mind_loss = 0.5 * loss_func[1](y_source, target) + 0.5 * loss_func[1](y_target, source)
        #     # multi-scale(y_true, y_pred)
        #     dice_loss = 0.5 * loss_func[2](y_source_mask, target_mask) + 0.5 * loss_func[2](y_target_mask, source_mask)
        #     # dice_loss = 0.5 * loss_func[2](target_mask, y_source_mask) + 0.5 * loss_func[2](source_mask, y_target_mask)
        #     dtmse_loss = 0.5 * loss_func[4](trans_source_dt, target_dt) + 0.5 * loss_func[4](trans_target_dt, source_dt)
        #     ngf_loss = 0.5 * loss_func[5](y_source, target) + 0.5 * loss_func[5](y_target, source)
        # else:
        #     ncc_loss = loss_func[0](y_source, target)
        #     mind_loss = loss_func[1](y_source, target)
        #     dice_loss = loss_func[2](y_source_mask, target_mask)
        #     # dice_loss = loss_func[2](target_mask, y_source_mask)
        #     dtmse_loss = loss_func[4](trans_source_dt, target_dt)
        #     ngf_loss = loss_func[5](y_source, target)

        if phase == "train" and bidir:
            ncc_loss = 0.5 * loss_func[0](y_source, target) + 0.5 * loss_func[0](y_target, source)
            mind_loss = 0.5 * loss_func[1](y_source, target) + 0.5 * loss_func[1](y_target, source)
            dice_loss = 0.5 * loss_func[2](y_source_mask, target_mask) + 0.5 * loss_func[2](source_mask, y_target_mask)
            ngf_loss = 0.5 * loss_func[5](y_source, target) + 0.5 * loss_func[5](y_target, source)
        else:
            ncc_loss = loss_func[0](y_source, target)
            mind_loss = loss_func[1](y_source, target)
            dice_loss = loss_func[2](y_source_mask, target_mask)
            ngf_loss = loss_func[5](y_source, target)

        grad_loss = loss_func[3](0, flow)

        loss = 0
        loss = loss_weights[0] * ncc_loss + \
               loss_weights[2] * dice_loss + \
               loss_weights[3] * grad_loss + \
               loss_weights[1] * mind_loss + \
               loss_weights[5] * ngf_loss

        sd_logjac = metric_func[0](disp_field.detach().cpu().numpy())
        # dist(y_pred, y), both are one-hot format
        # haus_dist
        metric_func[1](y_source_mask.detach().cpu(), target_mask.detach().cpu())
        # haus_dist = metric_func[1](y_source_mask.detach().cpu(), target_mask.detach().cpu())
        # surf_dist
        metric_func[2](y_source_mask.detach().cpu(), target_mask.detach().cpu())
        # surf_dist = metric_func[2](y_source_mask.detach().cpu(), target_mask.detach().cpu())
        # dice_score
        metric_func[3](y_source_mask.detach().cpu(), target_mask.detach().cpu())

        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()

        ncc_loss_list.append(ncc_loss.detach().cpu().numpy())
        mind_loss_list.append(mind_loss.detach().cpu().numpy())
        dice_loss_list.append(dice_loss.detach().cpu().numpy())
        grad_loss_list.append(grad_loss.detach().cpu().numpy())
        # dtmse_loss_list.append(dtmse_loss.detach().cpu().numpy())
        ngf_loss_list.append(ngf_loss.detach().cpu().numpy())
        epoch_loss_list.append(loss.detach().cpu().item())
        epoch_step_time.append(time.time() - step_start_time)
        sd_logjac_list.append(sd_logjac)
        # haus_dist_list.append(haus_dist.detach().cpu().mean().numpy())
        # surf_dist_list.append(surf_dist.detach().cpu().mean().numpy())
        # dice_score_list.append(dice_score.detach().cpu().mean().numpy())

        torch.cuda.empty_cache()

    logWriter.loss_per_epoch('ncc_loss', ncc_loss_list, phase, epoch_id)
    logWriter.loss_per_epoch('mind_loss', mind_loss_list, phase, epoch_id)
    logWriter.loss_per_epoch('dice_loss', dice_loss_list, phase, epoch_id)
    logWriter.loss_per_epoch('grad_loss', grad_loss_list, phase, epoch_id)
    logWriter.loss_per_epoch('dtmse_loss', dtmse_loss_list, phase, epoch_id)
    logWriter.loss_per_epoch('ngf_loss', ngf_loss_list, phase, epoch_id)
    logWriter.loss_per_epoch('losses', epoch_loss_list, phase, epoch_id)
    logWriter.time_per_epoch(epoch_step_time, phase, epoch_id)
    logWriter.loss_per_epoch('sd_logjac', sd_logjac_list, phase, epoch_id)
    logWriter.loss_per_epoch('haus_dist', metric_func[1].aggregate().item(), phase, epoch_id)
    logWriter.loss_per_epoch('surf_dist', metric_func[2].aggregate().item(), phase, epoch_id)
    logWriter.loss_per_epoch('dice_score', metric_func[3].aggregate().item(), phase, epoch_id)

    prediction = {'image': y_source, 'label': y_source_mask, 'flow': flow}
    logWriter.plot_per_epoch(data, prediction,
                             ncc_loss_list[-1], mind_loss_list[-1],
                             dice_loss_list[-1],
                             # dtmse_loss_list[-1],
                             grad_loss_list[-1], epoch_loss_list[-1],
                             phase, epoch_id)
    return np.mean(dice_loss_list)
