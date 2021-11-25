from monai.networks.nets import LocalNet
from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

from voxelmorph.torch import layers
from voxelmorph.torch.modelio import LoadableModel, store_config_args
from voxelmorph.torch.layers import SpatialTransformer, VecInt


class RegNet(LoadableModel):
    @store_config_args
    def __init__(self,
                 inshape=(64, 128, 128),
                 in_channels=2,
                 num_channel_initial=32,
                 extract_levels=[0, 1, 2, 3],
                 out_kernel_initializer="kaiming_uniform",
                 out_activation=None,
                 out_channels=3,
                 pooling=True,
                 concat_skip=False,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 half_out=False
                 ):
        """
        Parameters:
            localnet_config: Configurations for local net, dict
            inshape: Input shape. e.g. (64, 128, 128)
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            half_out: Whether LocalNet output fullsize or half-size  DVF/DDF when int_downsize=2.
                Default is False.
        """
        super().__init__()

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.localnet_model = LocalNet(spatial_dims=ndims,
                                       in_channels=in_channels,
                                       num_channel_initial=num_channel_initial,
                                       extract_levels=extract_levels,
                                       out_kernel_initializer=out_kernel_initializer,
                                       out_activation=out_activation,
                                       out_channels=out_channels,
                                       pooling=pooling,
                                       concat_skip=concat_skip)

        # configure optional resize layers (downsize)
        if int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        # Warning: nearest mode returns zero gradient
        # https://github.com/Project-MONAI/MONAI/discussions/2881
        self.mask_transformer = layers.SpatialTransformer(inshape, mode='nearest')

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                source_mask: torch.Tensor, target_mask: torch.Tensor,
                # source_dt: torch.Tensor, target_dt: torch.Tensor,
                registration: bool = False):
        '''
        Parameters:
            source: Source image tensor. Moving image tensor
            target: Target image tensor. Fixed image tensor
            source_mask: Source label tensor. Moving label tensor
            target_mask: Target label tensor. Fixed label tensor
            # source_dt: distance transform of source label
            # target_dt: distance transform of target label
            registration: Return transformed image and flow. Default is False.
        '''
        x = torch.cat([source, target], dim=1)
        flow_field = self.localnet_model(x)

        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        y_source = self.transformer(source, pos_flow)
        if not registration:
            # use linear resampling to enable non-zero gradient
            y_source_mask = self.transformer(source_mask, pos_flow)
            # trans_source_dt = self.transformer(source_dt, pos_flow)
        else:
            y_source_mask = self.mask_transformer(source_mask, pos_flow)
            # trans_source_dt = self.mask_transformer(source_dt, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None
        y_target_mask = self.transformer(target_mask, neg_flow) if self.bidir else None
        # trans_target_dt = self.transformer(target_dt, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_source_mask, y_target, y_target_mask,
                    preint_flow, pos_flow) if self.bidir else \
                (y_source, y_source_mask, preint_flow, pos_flow)
        else:
            return y_source, y_source_mask, pos_flow

    def inference(self, source: torch.Tensor, target: torch.Tensor, source_mask: torch.Tensor):
        x = torch.cat([source, target], dim=1)
        flow_field = self.localnet_model(x)
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            if self.fullsize:
                full_pos_flow = self.fullsize(pos_flow)
        y_source_mask = self.mask_transformer(source_mask, full_pos_flow)
        return y_source_mask, pos_flow
