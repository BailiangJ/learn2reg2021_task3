import os
import numpy as np
import torch
import gc
import time
import ml_collections

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from data_utils import load_data_task03
from voxelmorph.torch.losses import NCC, Grad
from monai.losses import DiceLoss, MultiScaleLoss
from losses import DTMSELoss, MINDSSCLoss
from normalized_gradient_field import NormalizedGradientField3d
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric, DiceMetric
from monai.networks import one_hot
from monai.data import DataLoader
from metrics import SDlogJac
from deepregnet import RegNet
from log_utils import LogWriter
from utils import run_epoch
from my_argparse import regnet_argparse
from torchinfo import summary
import polyaxon_helper
from monai.utils import first

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.backends.cudnn.deterministric = True
torch.backends.cudnn.benchmark = True
arg = regnet_argparse(config_files="deepregnet.ini")

out_path = polyaxon_helper.get_outputs_path()
model_dir = os.path.join(out_path, arg.model_dir)
os.makedirs(model_dir, exist_ok=True)
log_dir = os.path.join(out_path, arg.log_dir)
logWriter = LogWriter(log_dir)

dataset = load_data_task03(arg.data_dir, cache_rate=0.05, num_workers=4)
train_dataset = dataset[:-20]
val_dataset = dataset[-20:]
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
train_size = len(train_loader)
val_size = len(val_loader)


def define_model(arg):
    config = ml_collections.ConfigDict()
    config.spatial_dims = len(arg.inshape)
    config.in_channels = 2
    config.out_channels = 3
    config.num_channel_initial = arg.num_channel_initial
    config.extract_levels = arg.extract_levels
    config.out_activation = None
    config.out_kernel_initializer = "zeros"
    config.pooling = True
    config.concat_skip = False
    if not arg.use_last_ckpt:
        model = RegNet(inshape=arg.inshape,
                       in_channels=config.in_channels,
                       num_channel_initial=config.num_channel_initial,
                       extract_levels=config.extract_levels,
                       out_kernel_initializer=config.out_kernel_initializer,
                       out_activation=config.out_activation,
                       out_channels=config.out_channels,
                       pooling=config.pooling,
                       concat_skip=config.concat_skip,
                       int_steps=arg.int_steps,
                       int_downsize=arg.int_downsize,
                       bidir=arg.bidir)
    else:
        model = RegNet.load(arg.load_model, arg.device)
    model = model.to(arg.device)
    with torch.no_grad():
        summary(model, [(1, 1, *arg.inshape), (1, 1, *arg.inshape),
                        (1, 1, *arg.inshape), (1, 1, *arg.inshape),
                        (1, 1, *arg.inshape), (1, 1, *arg.inshape)])
    torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.get_device_name(0), torch.cuda.device_count())
    return model, config


model, config = define_model(arg)
if arg.use_last_ckpt:
    start_epoch = arg.start_epoch
    # lr = arg.lr * (arg.decay_rate ** (start_epoch - 1))
    lr = arg.lr
    model.bidir = arg.bidir
else:
    start_epoch = 0
    lr = arg.lr

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=arg.decay_rate)

loss_func = [NCC().loss,
             MINDSSCLoss(radius=2,
                         dilation=2),
             MultiScaleLoss(DiceLoss(include_background=False, to_onehot_y=False),
                            scales=[0]),
             Grad('l2', loss_mult=arg.int_downsize).loss,
             DTMSELoss(alpha=2),
             NormalizedGradientField3d(eps_src=1e-3, eps_tar=1e-3, mm_spacing=1)]

loss_weights = [arg.ncc_loss_weight, arg.mind_loss_weight, arg.dice_loss_weight,
                arg.grad_loss_weight, arg.dtmse_loss_weight, arg.ngf_loss_weight]

metric_func = [SDlogJac(),
               HausdorffDistanceMetric(percentile=95),
               SurfaceDistanceMetric(),
               DiceMetric(include_background=False)]

args_dict = vars(arg)
args_dict.update(dict(config))
args_dict['train_size'] = train_size
args_dict['val_size'] = val_size
logWriter.log_configuration(args_dict)

# logWriter.log_configuration(bidir=arg.bidir, int_steps=arg.int_steps, int_downsize=arg.int_downsize,
#                             num_channel_initial=arg.num_channel_initial, extract_levels=arg.extract_levels,
#                             out_kernel_initializer=config.out_kernel_initializer, pooling=config.pooling,
#                             concat_skip=config.concat_skip,
#                             lr=arg.lr, decay_rate=arg.decay_rate,
#                             ncc_loss_weight=arg.ncc_loss_weight,
#                             mind_loss_weight=arg.mind_loss_weight,
#                             dice_loss_weight=arg.dice_loss_weight,
#                             grad_loss_weight=arg.grad_loss_weight,
#                             dtmse_loss_weight=arg.dtmse_loss_weight,
#                             ngf_loss_weight=arg.ngf_loss_weight,
#                             flipping=arg.flipping)

best_dice_loss = 1
for epoch in range(start_epoch, arg.max_epochs):
    if epoch % arg.val_interval == 0 or epoch == start_epoch:
        model.eval()
        phase = 'val'
        with torch.no_grad():
            val_dice_loss = run_epoch(model, val_loader, optimizer,
                                      loss_func, loss_weights, metric_func,
                                      arg.bidir, arg.flipping, logWriter,
                                      arg.device, phase, epoch)
        if (arg.max_epochs - epoch) < arg.eval_best_epoch and val_dice_loss < best_dice_loss:
            best_dice_loss = val_dice_loss
            model.save(os.path.join(model_dir, 'best.pt'))
            print(f"save best model at {epoch} epoch")

    model.train()
    phase = 'train'
    run_epoch(model, train_loader, optimizer,
              loss_func, loss_weights, metric_func,
              arg.bidir, arg.flipping, logWriter,
              arg.device, phase, epoch)
    lr_scheduler.step()

    if epoch % arg.save_interval == 0:
        model.save(os.path.join(model_dir, f'{epoch:04d}.pt'))

model.save(os.path.join(model_dir, f'{arg.max_epochs:04d}.pt'))
