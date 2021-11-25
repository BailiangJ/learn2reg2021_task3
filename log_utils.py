import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import yaml


class LogWriter(object):
    def __init__(self, log_dir_name):
        train_log_path, val_log_path, eval_log_path = os.path.join(log_dir_name, "train"), \
                                                      os.path.join(log_dir_name, "val"), \
                                                      os.path.join(log_dir_name, "eval")
        self.writer = {"train": SummaryWriter(train_log_path),
                       "val": SummaryWriter(val_log_path),
                       # "eval": SummaryWriter(eval_log_path)
                       }
        self.log_dir_name = log_dir_name

    def loss_per_epoch(self, loss_type, loss_arr, phase, epoch):
        loss = np.mean(loss_arr)
        self.writer[phase].add_scalar(loss_type + '/per_epoch', loss, epoch)
        print(f'{phase} / epoch {epoch:03d} / {loss_type} = {loss:.5f}')

    def plot_per_epoch(self, data, prediction, ncc_loss,
                       mind_loss, dice_loss,
                       # mse_loss,
                       grad_loss,
                       loss, phase, epoch):
        source_image = data["image"][0][0].detach().cpu().numpy()
        source_label = data["label"][0][0].detach().cpu().numpy()
        target_image = data["image"][1][0].detach().cpu().numpy()
        target_label = data["label"][1][0].detach().cpu().numpy()
        pred_image = prediction["image"][0][0].detach().cpu().numpy()
        pred_label = prediction["label"][0][0].detach().cpu().numpy()
        pred_flow = prediction["flow"][0].detach().cpu().numpy().transpose(1, 2, 3, 0)
        shape = source_image.shape
        plt.set_cmap('gray')
        fig = plt.figure(figsize=(12, 4), dpi=180, facecolor='w', edgecolor='k')
        fig.suptitle(f'ncc_loss:{ncc_loss:.2f},'
                     f'mind_loss:{mind_loss:.2f},'
                     f'dice_loss:{dice_loss:.2f},'
                     f'grad_loss:{grad_loss:.2f}')
        num_plots = 7
        ax = fig.add_subplot(1, num_plots, 1)
        ax.set_title("moving image")
        ax.imshow(source_image[shape[0] // 2, :, :])
        ax = fig.add_subplot(1, num_plots, 2)
        ax.set_title("moving label")
        ax.imshow(source_label[shape[0] // 2, :, :])
        ax = fig.add_subplot(1, num_plots, 3)
        ax.set_title("fixed image")
        ax.imshow(target_image[shape[0] // 2, :, :])
        ax = fig.add_subplot(1, num_plots, 4)
        ax.set_title("fixed label")
        ax.imshow(target_label[shape[0] // 2, :, :])
        ax = fig.add_subplot(1, num_plots, 5)
        ax.set_title("warpped label")
        ax.imshow(pred_label[shape[0] // 2, :, :])
        ax = fig.add_subplot(1, num_plots, 6)
        ax.set_title("warpped image")
        ax.imshow(pred_image[shape[0] // 2, :, :])
        ax = fig.add_subplot(1, num_plots, 7)
        ax.set_title("flow")
        flow_shape = pred_flow.shape
        self.plot_flow(ax, pred_flow[flow_shape[0] // 2, ::2, ::2, 1:])
        self.writer[phase].add_figure(f'{phase} / epoch {epoch:03d}', fig)

    def plot_flow(self, ax, pred_flow, img_indexing=True, quiver_width=None, scale=1):
        if img_indexing:
            pred_flow = np.flipud(pred_flow)
        ax.set_title("pred flow")

        u, v = pred_flow[..., 0], pred_flow[..., 1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        colormap = cm.winter

        ax.quiver(u, v,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  width=quiver_width,
                  scale=scale
                  )
        ax.axis('equal')
        ax.axis('off')

    def time_per_epoch(self, epoch_step_time, phase, epoch):
        step_time = np.mean(epoch_step_time)
        epoch_time = np.sum(epoch_step_time)
        print(f'{phase} / epoch {epoch:03d} / {step_time:0.4f} sec/step / {epoch_time:0.4f} sec/epoch')

    def log_configuration(self, **kwargs):
        with open(os.path.join(self.log_dir_name, "configs.yaml"), 'w') as yamlfile:
            yaml.dump(kwargs, yamlfile)
