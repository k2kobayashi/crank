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

import numpy as np
import soundfile as sf
import torch
from crank.net.module.mlfb import LogMelFilterBankLayer
from crank.net.module.sinc_conv import SincConvPreprocessingLayer
from crank.utils import load_yaml

B, T = 1, 65536
datadir = Path(__file__).parent / "data"
ymlf = datadir / "mlfb_vqvae_22050.yml"
spkrymlf = datadir / "spkr.yml"

conf = load_yaml(ymlf)
wavf = datadir / "SF1_10001.wav"


def test_sincconv():
    sinc_conv = SincConvPreprocessingLayer(
        in_channels=1,
        sincconv_channels=32,
        sincconv_kernel_size=65,
        out_channels=80,
        kernel_sizes=[4, 4, 4, 2],
    )
    x, fs = sf.read(str(wavf))
    x = np.array(x, dtype=np.float32)

    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(-1)
    y = sinc_conv(x)

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
    )
    y_mlfb = mlfb_layer(x.squeeze(-1))

    assert y.size() == y_mlfb.size()
