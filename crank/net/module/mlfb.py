#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import librosa
import scipy.signal
import torch
import torch.nn as nn


class MLFBLayer(torch.nn.Module):
    def __init__(
        self, fs=22050, fft_size=1024, n_mels=80, fmin=None, fmax=None, eps=1.0e-10
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        mel_basis = librosa.filters.mel(
            sr=fs,
            n_fft=fft_size,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.eps = eps
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis.T).float())

    def forward(
        self,
        x,
    ):
        mlfb = torch.matmul(x, self.mel_basis)
        mlfb = torch.clamp(mlfb, min=self.eps).log10()
        return mlfb


class STFTLayer(torch.nn.Module):
    def __init__(
        self,
        fs=22050,
        hop_size=256,
        fft_size=1024,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        return_complex=False,
    ):
        super().__init__()
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.win_length = fft_size if win_length is None else win_length
        self.center = center
        self.pad_mode = pad_mode
        self.return_complex = return_complex
        """
        prepare window parameter type of window
        - "hann": hanning window
        - "param": parameter-based window
        - "conv": convolution-based window
        """
        self.window_type = window
        if window == "param":
            win = scipy.signal.get_window("hann", self.win_length).astype(float)
            self.register_parameter(
                "window", nn.Parameter(torch.from_numpy(win), requires_grad=True)
            )
        elif window == "conv":
            kernel_size = 65
            self.window_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=24,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.Sigmoid(),
            )
        else:
            self.window = window

    def forward(self, x):
        if self.window_type == "param":
            window = self.window
        elif self.window_type == "conv":
            x = x.unsqueeze(-1).transpose(1, 2)
            x = torch.mean(self.window_conv(x).transpose(1, 2), -1)
            window = None
        else:
            f = getattr(torch, f"{self.window}_window")
            window = f(self.win_length, dtype=x.dtype, device=x.device)

        stft = torch.stft(
            x,
            n_fft=self.fft_size,
            win_length=self.win_length,
            hop_length=self.hop_size,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            return_complex=self.return_complex,
        )
        return stft.transpose(1, 2).float()


class MLFBScalerLayer(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        print(scaler.var_, scaler.mean_)

        print(torch.from_numpy(scaler.mean_).float().size())
        self.register_parameter(
            "mean",
            nn.Parameter(torch.from_numpy(scaler.mean_).float(), requires_grad=False),
        )
        self.register_parameter(
            "std",
            nn.Parameter(
                torch.from_numpy(scaler.var_).float().sqrt(), requires_grad=False
            ),
        )

    def forward(self, x):
        return (x - self.mean) / self.std


class LogMelFilterBankLayer(nn.Module):
    def __init__(
        self,
        fs=22050,
        hop_size=256,
        fft_size=1024,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        n_mels=80,
        fmin=None,
        fmax=None,
        scaler=None,
    ):
        super().__init__()
        self.stft_layer = STFTLayer(
            fs,
            hop_size,
            fft_size,
            win_length,
            window,
            center=center,
            pad_mode=pad_mode,
        )
        self.mlfb_layer = MLFBLayer(fs, fft_size, n_mels, fmin, fmax)
        if scaler is not None:
            self.scaler_layer = MLFBScalerLayer(scaler)
        else:
            self.scaler_layer = None

    def forward(self, x):
        stft = self.stft_layer(x)
        amplitude = torch.sqrt(stft[..., 0] ** 2 + stft[..., 1] ** 2)
        mlfb = self.mlfb_layer(amplitude)
        if self.scaler_layer is not None:
            mlfb = self.scaler_layer(mlfb)
        return mlfb
