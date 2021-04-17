#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch
from parallel_wavegan.bin.preprocess import logmelfilterbank
from sklearn.preprocessing import StandardScaler

from crank.net.module.mlfb import LogMelFilterBankLayer, STFTLayer
from crank.utils import load_yaml, plot_mlfb

datadir = Path(__file__).parent / "data"
ymlf = datadir / "mlfb_vqvae_22050.yml"
spkrymlf = datadir / "spkr.yml"

conf = load_yaml(ymlf)
wavf = datadir / "SF1_10001.wav"


@pytest.mark.parametrize(
    "window",
    [("hann"), ("param"), ("conv")],
    ids=["hann_window", "param_window", "conv_window"],
)
def test_feature_onthefly(window):
    x, fs = sf.read(str(wavf))
    x = np.array(x, dtype=np.float)

    # extract feature
    mlfb_np = logmelfilterbank(
        x,
        fs,
        hop_size=conf["feature"]["hop_size"],
        fft_size=conf["feature"]["fftl"],
        win_length=conf["feature"]["win_length"],
        window="hann",
        num_mels=conf["feature"]["mlfb_dim"],
        fmin=conf["feature"]["fmin"],
        fmax=conf["feature"]["fmax"],
    )

    # extract feature by pytorch
    mlfb_layer = LogMelFilterBankLayer(
        fs=fs,
        hop_size=conf["feature"]["hop_size"],
        fft_size=conf["feature"]["fftl"],
        win_length=conf["feature"]["win_length"],
        window=window,
        center=False,
        n_mels=conf["feature"]["mlfb_dim"],
        fmin=conf["feature"]["fmin"],
        fmax=conf["feature"]["fmax"],
    )

    raw = torch.from_numpy(x).unsqueeze(0).float()
    mlfb_torch = mlfb_layer(raw).detach().squeeze(0).detach().cpu().numpy()

    plot_mlfb(mlfb_torch, datadir / "mlfb_torch.png")
    plot_mlfb(mlfb_np, datadir / "mlfb_np.png")

    if window == "han":
        np.testing.assert_equal(mlfb_np.shape, mlfb_torch.shape)
        np.testing.assert_almost_equal(mlfb_np, mlfb_torch, decimal=4)


def test_feature_onthefly_padding():
    """check equivalent when discarding some frames like dataloader"""
    x, fs = sf.read(str(wavf))
    x = np.array(x, dtype=np.float)
    mlfb_layer = LogMelFilterBankLayer(
        fs=fs,
        hop_size=conf["feature"]["hop_size"],
        fft_size=conf["feature"]["fftl"],
        win_length=conf["feature"]["win_length"],
        window="hann",
        center=False,
        n_mels=conf["feature"]["mlfb_dim"],
        fmin=conf["feature"]["fmin"],
        fmax=conf["feature"]["fmax"],
    )
    batch_len = conf["batch_len"]
    fftl = conf["feature"]["fftl"]
    hop_size = conf["feature"]["hop_size"]

    # create reference mel-filterbank
    mlfb_np = logmelfilterbank(
        x,
        fs,
        hop_size=conf["feature"]["hop_size"],
        fft_size=conf["feature"]["fftl"],
        win_length=conf["feature"]["win_length"],
        window="hann",
        num_mels=conf["feature"]["mlfb_dim"],
        fmin=conf["feature"]["fmin"],
        fmax=conf["feature"]["fmax"],
    )
    p = np.random.choice(range(10, 500, 1))
    mlfb_np = mlfb_np[p : p + batch_len]
    assert p * hop_size - int(fftl // 2) >= 0
    x_mod = x[
        p * hop_size
        - int(fftl // 2) : p * hop_size
        + hop_size * batch_len
        - 1
        + int(fftl // 2)
    ]

    # extract feature by pytorch
    raw = torch.from_numpy(x_mod).unsqueeze(0).float()
    mlfb_torch = mlfb_layer(raw).squeeze(0).detach().cpu().numpy()

    np.testing.assert_equal(mlfb_np.shape, mlfb_torch.shape)
    np.testing.assert_almost_equal(mlfb_np, mlfb_torch, decimal=3)


def test_stft_torch():
    wavf = datadir / "SF1_10001.wav"
    x, fs = sf.read(str(wavf))

    # test stft
    stft_np = librosa.stft(
        x,
        n_fft=conf["feature"]["fftl"],
        hop_length=conf["feature"]["hop_size"],
        win_length=conf["feature"]["win_length"],
        window="hann",
        pad_mode="reflect",
    )
    spc_np = np.abs(stft_np).T
    stft_layer = STFTLayer(
        fs=fs,
        hop_size=conf["feature"]["hop_size"],
        fft_size=conf["feature"]["fftl"],
        win_length=conf["feature"]["win_length"],
        window="hann",
        center=True,
    )
    stft = stft_layer(torch.from_numpy(x).unsqueeze(0))
    spc_torch = (
        torch.sqrt(stft[..., 0] ** 2 + stft[..., 1] ** 2)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    plot_mlfb(spc_torch, datadir / "spc_torch.png")
    plot_mlfb(spc_np, datadir / "spc_np.png")

    np.testing.assert_equal(spc_np.shape, spc_torch.shape)
    np.testing.assert_almost_equal(spc_np, spc_torch, decimal=5)


def test_feature_onthefly_scaler():
    x, fs = sf.read(str(wavf))
    x = np.array(x, dtype=np.float)

    # extract feature
    mlfb_np = logmelfilterbank(
        x,
        fs,
        hop_size=conf["feature"]["hop_size"],
        fft_size=conf["feature"]["fftl"],
        win_length=conf["feature"]["win_length"],
        window="hann",
        num_mels=conf["feature"]["mlfb_dim"],
        fmin=conf["feature"]["fmin"],
        fmax=conf["feature"]["fmax"],
    )

    ss = StandardScaler()
    ss.partial_fit(mlfb_np)

    # extract feature by pytorch
    mlfb_layer = LogMelFilterBankLayer(
        fs=fs,
        hop_size=conf["feature"]["hop_size"],
        fft_size=conf["feature"]["fftl"],
        win_length=conf["feature"]["win_length"],
        window="hann",
        center=True,
        n_mels=conf["feature"]["mlfb_dim"],
        fmin=conf["feature"]["fmin"],
        fmax=conf["feature"]["fmax"],
        scaler=ss,
    )

    raw = torch.from_numpy(x).unsqueeze(0).float()
    mlfb_torch = mlfb_layer(raw).detach().squeeze(0).detach().cpu().numpy()

    plot_mlfb(mlfb_torch, datadir / "mlfb_torch_scaler.png")
    plot_mlfb(mlfb_np, datadir / "mlfb_np_scaler.png")

    np.testing.assert_equal(mlfb_np.shape, mlfb_torch.shape)
    np.testing.assert_almost_equal(ss.transform(mlfb_np), mlfb_torch, decimal=3)
