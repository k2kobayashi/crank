#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
"""Speaker advarsarial network.

"""

import torch
import torch.nn as nn
from parallel_wavegan.models import (
    ParallelWaveGANDiscriminator,
    ResidualParallelWaveGANDiscriminator,
)


class SpeakerAdversarialNetwork(nn.Module):
    def __init__(self, conf, spkr_size=0):
        super(SpeakerAdversarialNetwork, self).__init__()
        self.conf = conf
        self.spkr_size = spkr_size
        self._construct_net()

    def forward(self, x, detach=False):
        x = torch.cat(x, axis=-1)
        if detach:
            x = x.detach()
        x = self.grl(x).transpose(1, 2)
        x = self.classifier(x).transpose(1, 2)
        return x

    def _construct_net(self):
        self.grl = GradientReversalLayer(scale=self.conf["spkradv_lambda"])

        # TODO: investigate peformance of residual network
        # if self.conf["use_residual_network"]:
        #     self.classifier = ResidualParallelWaveGANDiscriminator(
        #         in_channels=sum(
        #             self.conf["emb_dim"][:self.conf["n_vq_stacks"]]),
        #         out_channels=self.spkr_size,
        #         kernel_size=self.conf["spkradv_kernel_size"],
        #         layers=self.conf["n_spkradv_layers"],
        #         stacks=self.conf["n_spkradv_layers"] // 2,
        #     )
        # else:
        self.classifier = ParallelWaveGANDiscriminator(
            in_channels=sum(self.conf["emb_dim"][: self.conf["n_vq_stacks"]]),
            out_channels=self.spkr_size,
            kernel_size=self.conf["spkradv_kernel_size"],
            layers=self.conf["n_spkradv_layers"],
            conv_channels=64,
            dilation_factor=1,
            nonlinear_activation="LeakyReLU",
            nonlinear_activation_params={"negative_slope": 0.2},
            bias=True,
            use_weight_norm=True,
        )


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_forward, scale):
        ctx.save_for_backward(scale)
        return input_forward

    @staticmethod
    def backward(ctx, grad_backward):
        (scale,) = ctx.saved_tensors
        return scale * -grad_backward, None


class GradientReversalLayer(nn.Module):
    def __init__(self, scale=1.0):
        super(GradientReversalLayer, self).__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.scale)
