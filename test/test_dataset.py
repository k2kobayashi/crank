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

import joblib
import pytest
import torch
from torch.utils.data import DataLoader

from crank.net.trainer.dataset import BaseDataset
from crank.utils import load_yaml, open_scpdir

datadir = Path(__file__).parent / "data"
ymlf = datadir / "mlfb_vqvae_22050.yml"
h5f = datadir / "SF1" / "SF1_10001.feats.h5"
scalerf = datadir / "scaler.pkl"

B, T, D = 3, 100, 20
scaler = joblib.load(scalerf)


@pytest.mark.parametrize(
    "decoder_f0, use_mcep, use_raw",
    [
        (True, False, False),
        (False, False, False),
        (False, True, False),
        (True, False, True),
    ],
    ids=["f0condition", "no_f0condition", "use_mcep", "use_raw"],
)
def test_dataset(decoder_f0, use_mcep, use_raw):
    conf = load_yaml(ymlf)
    conf["decoder_f0"] = decoder_f0
    conf["receptive_size"] = 128
    if use_mcep:
        conf["input_feat_type"] = "mcep"
        conf["output_feat_type"] = "mcep"
        conf["ignore_scaler"] = ["mcep", "raw"]
    if use_raw:
        conf["use_raw"] = True
        conf["input_feat_type"] = "mlfb"
        conf["ignore_scaler"] = ["raw"]

    scp = {}
    scpdir = datadir / "scpdir"
    for phase in ["train", "dev", "eval"]:
        scp[phase] = open_scpdir(scpdir / phase)
        scp[phase]["feats"] = {"01": h5f, "02": h5f, "03": h5f}
    dataset = BaseDataset(conf, scp, phase="train", scaler=scaler)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=1)

    for i, batch in enumerate(dataloader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                pass
                # print(k, v.type(), v.size())
            else:
                pass
                # print(k, v, type(v))
