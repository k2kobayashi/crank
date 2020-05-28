#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.


"""
Multi-resolution STFT Loss

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_size, window):
    """Return stft magnitude spectral

    Args:
        x : input tensor (B, T, D)
    """
    x = x.transpose(1, 2).reshape(-1, x.size(1))
    x_stft = torch.stft(x, fft_size, win_size, hop_size, window)
    real, imag = x_stft[..., 0], x_stft[..., 1]
    y = torch.clamp(real ** 2 + imag ** 2, min=1e-7).transpose(2, 1)
    return torch.sqrt(y)


class STFTLoss(nn.Module):
    def __init__(self, fft_size=32, win_size=20, hop_size=10, logratio=0.0):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.win_size = win_size
        self.hop_size = hop_size
        self.logratio = logratio
        self.window = torch.hann_window(win_size)

    def forward(self, x, y):
        """
        Args:
            x, y: input Tensor (B, T, D)
        """
        x_mag = stft(x, self.fft_size, self.win_size, self.hop_size, self.window)
        y_mag = stft(y, self.fft_size, self.win_size, self.hop_size, self.window)

        mag_loss = F.l1_loss(x_mag, y_mag)
        lmag_loss = F.l1_loss(x_mag.log(), y_mag.log())
        loss = (1 - self.logratio) * mag_loss + self.logratio * lmag_loss
        return loss


class MultiSizeSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[32, 128, 256],
        win_sizes=[20, 80, 160],
        hop_sizes=[10, 20, 30],
        logratio=0.0,
    ):
        super(MultiSizeSTFTLoss, self).__init__()
        self.loss_layers = torch.nn.ModuleList()
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes, hop_sizes):
            self.loss_layers.append(
                STFTLoss(fft_size, hop_size, win_size, logratio=logratio)
            )

    def forward(self, x, y):
        """
        Args:
            x, y: input Tensor (B, T, D)
        """
        losses = []
        for layer in self.loss_layers:
            loss = layer(x, y)
            losses.append(loss)
        loss = sum(losses) / len(losses)
        return loss
