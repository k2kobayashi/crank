#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""


from pathlib import Path

from crank.feature import Feature
from crank.utils import load_yaml

datadir = Path(__file__).parent / "data"


def test_feature():
    conf = load_yaml(datadir / "mlfb_vqvae.yml")
    spkr_conf = load_yaml(datadir / "spkr.yml")
    feat = Feature(datadir, conf["feature"], spkr_conf["SF1"], synth_flag=True)
    feat.analyze(
        datadir / "SF1_10001.wav", gl_flag=True,
    )
    (datadir / "SF1_10001.h5").unlink()
    (datadir / "SF1_10001_anasyn.wav").unlink()


def test_feature_8k():
    conf = load_yaml(datadir / "mlfb_vqvae.yml")
    conf["feature"].update(
        {
            "fs": 8000,
            "fftl": 512,
            "fmin": 80,
            "fmax": 3800,
            "hop_size": 80,
            "mlfb_dim": 80,
        }
    )
    spkr_conf = load_yaml(datadir / "spkr.yml")
    feat = Feature(datadir, conf["feature"], spkr_conf["SF1"], synth_flag=False)
    feat.analyze(datadir / "SF1_10001_8k.wav", gl_flag=True)
    (datadir / "SF1_10001_8k.h5").unlink()
