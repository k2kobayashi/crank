#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# test_loss.py
#   First ver.: 2020-05-12
#
#   Copyright 2020
#       K. Kobayashi <root.4mac@gmail.com>
#
#   Distributed under terms of the MIT license.
#

"""


"""

import torch
from crank.net.module.loss import CustomFeatureLoss, MultiSizeSTFTLoss, STFTLoss

B, T, D = 3, 1000, 10


def test_stftloss():
    fft_size, hop_size, win_size = 32, 10, 20
    x = torch.randn((B, T * 100, D))
    y = torch.randn((B, T * 100, D))
    criterion = STFTLoss(
        fft_size=fft_size,
        hop_size=hop_size,
        win_size=win_size,
        logratio=0.0,
        device="cpu",
    )
    _ = criterion(x, y)
    criterion = MultiSizeSTFTLoss(device="cpu")
    _ = criterion(x, y)


def test_customloss():
    x = torch.randn((B, T, D))
    y = torch.randn((B, T, D))
    mask = x.ge(0)
    for c in [-8, -2, 0, 2, 8]:
        for loss_type in ["l1", "mse", "stft"]:

            criterion = CustomFeatureLoss(
                loss_type=loss_type, causal=True, device="cpu"
            )
            if loss_type != "stft":
                _ = criterion(x, y, mask=mask, causal_size=c)
            else:
                _ = criterion(x, y)
