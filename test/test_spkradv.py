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
import torch

from crank.utils import load_yaml
from crank.net.module.spkradv import SpeakerAdversarialNetwork

datadir = Path(__file__).parent / "data"
ymlf = datadir / "mlfb_vqvae_22050.yml"
conf = load_yaml(ymlf)
B, T = 3, 100


def test_spkradv():
    net = SpeakerAdversarialNetwork(conf=conf, spkr_size=32)
    x = [torch.randn((B, T, conf["emb_dim"][d])) for d in range(conf["n_vq_stacks"])]
    _ = net(x)
