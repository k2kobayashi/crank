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
from crank.net.module.loss import MultiSizeSTFTLoss, STFTLoss

B, T, D = 3, 1000, 80


def test_stftloss():
    fft_size = 32
    hop_size = 10
    win_size = 20

    x = torch.randn((B, T, D))
    y = torch.randn((B, T, D))

    criterion = STFTLoss(
        fft_size=fft_size, hop_size=hop_size, win_size=win_size, logratio=0.
    )
    loss = criterion(x, y)

    criterion = MultiSizeSTFTLoss()
    loss = criterion(x, y)
