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
import soundfile as sf
import torch
from crank.net.module.mlfb import LogMelFilterBankLayer, MLFBLayer, STFTLayer
from crank.utils import load_yaml, plot_mlfb

datadir = Path(__file__).parent / "data"
ymlf = datadir / "mlfb_vqvae_22050.yml"
spkrymlf = datadir / "spkr.yml"

# extract feature by librosa
conf = load_yaml(ymlf)

wavf = datadir / "SF1_10001.wav"


def test_feature_onthefly():
    # extract numpy
    from parallel_wavegan.bin.preprocess import logmelfilterbank

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
        window="hann",
        n_mels=conf["feature"]["mlfb_dim"],
        fmin=conf["feature"]["fmin"],
        fmax=conf["feature"]["fmax"],
    )
    mlfb_torch = mlfb_layer(torch.from_numpy(x).float()).cpu().numpy()

    plot_mlfb(mlfb_torch, datadir / "mlfb_torch.png")
    plot_mlfb(mlfb_np, datadir / "mlfb_np.png")

    np.testing.assert_equal(mlfb_np.shape, mlfb_torch.shape)
    np.testing.assert_almost_equal(mlfb_np, mlfb_torch, decimal=4)


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
    )
    stft = stft_layer(torch.from_numpy(x))
    spc_torch = torch.sqrt(stft[..., 0] ** 2 + stft[..., 1] ** 2).cpu().numpy()
    plot_mlfb(spc_torch, datadir / "spc_torch.png")
    plot_mlfb(spc_np, datadir / "spc_np.png")

    np.testing.assert_equal(spc_np.shape, spc_torch.shape)
    np.testing.assert_almost_equal(spc_np, spc_torch)
