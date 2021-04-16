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
from parallel_wavegan.bin.preprocess import logmelfilterbank

from crank.net.module.vqvae2 import VQVAE2
from crank.utils import load_yaml

SPKR_SIZE = 10
B, D = 3, 80

datadir = Path(__file__).parent / "data"
ymlf = datadir / "mlfb_vqvae_22050.yml"
spkrymlf = datadir / "spkr.yml"
conf = load_yaml(ymlf)
wavf = datadir / "SF1_10001.wav"


def test_vqvae():
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
    # NOTE: eigher discard frames or pad raw waveform for stft
    T = mlfb_np.shape[0] - 8

    conf["use_raw"] = True
    conf["input_feat_type"] = "mlfb"
    conf["ignore_scaler"] = ["raw"]
    conf["use_preprocessed_scaler"] = False
    fs = (conf["feature"]["fs"],)

    model = VQVAE2(conf, spkr_size=SPKR_SIZE)
    enc_h = None
    dec_h = torch.randn((B, T, 2))
    spkrvec = torch.ones((B, T)) * 5

    mlfb = torch.from_numpy(mlfb_np).float()  # noqa
    raw = torch.from_numpy(x).unsqueeze(0).float()
    _ = model.forward(raw, enc_h, dec_h, spkrvec=spkrvec.long())
    # y = model.cycle_forward(
    #     mlfb, enc_h, dec_h, enc_h, dec_h, spkrvec.long(), spkrvec.long()
    # )
