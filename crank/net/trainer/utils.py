#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. Kobayashi <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Utilities for trainer

"""

import torch_optimizer as toptim
from crank.net.module.loss import CustomFeatureLoss
from crank.net.trainer.dataset import BaseDataset, calculate_maxflen
from pytorch_lamb import Lamb
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def get_criterion(conf, device="cuda"):
    criterion = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "ce": nn.CrossEntropyLoss(ignore_index=-100),
        "kld": nn.KLDivLoss(reduction="mean"),
        "fmse": CustomFeatureLoss(loss_type="mse", causal_size=conf["causal_size"]),
        "fl1": CustomFeatureLoss(loss_type="l1", causal_size=conf["causal_size"]),
        "fstft": CustomFeatureLoss(
            loss_type="stft",
            causal_size=conf["causal_size"],
            stft_params=conf["stft_params"],
            device=device,
        ),
    }
    return criterion


def get_optimizer(conf, model):
    def return_optim(model, optim_type, lr):
        if optim_type == "adam":
            return optim.Adam(model.parameters(), lr=lr)
        elif optim_type == "radam":
            return toptim.RAdam(model.parameters(), lr=lr)
        elif optim_type == "lamb":
            return Lamb(model.parameters(), lr=lr)
        else:
            raise ValueError("Invalid optimizer type")

    optimizer = {}
    for m in ["G", "D", "C", "SPKRADV"]:
        if m in model:
            opt = return_optim(
                model[m], conf["optim"][m]["type"], conf["optim"][m]["lr"]
            )
            optimizer[m] = opt
    return optimizer


def get_scheduler(conf, optimizer):
    def return_scheduler(optim, step_size, gamma):
        return StepLR(optim, step_size=step_size, gamma=gamma)

    scheduler = {}
    for m in ["G", "D", "C", "SPKRADV"]:
        if m in optimizer:
            sche = return_scheduler(
                optimizer[m],
                conf["optim"][m]["decay_step_size"],
                conf["optim"][m]["decay_size"],
            )
            scheduler[m] = sche
    return scheduler


def get_dataloader(conf, scp, scaler, flag="train", n_jobs=10):
    if flag in ["train", "reconstruction"]:
        feats = list(scp["train"]["feats"].values()) + list(
            scp["dev"]["feats"].values()
        )
    elif flag in ["eval"]:
        feats = list(scp["eval"]["feats"].values())

    if flag in ["reconstruction", "eval"]:
        conf["batch_len"] = calculate_maxflen(feats)

    spkrs = dict(zip(scp["train"]["spkrs"], range(len(scp["train"]["spkrs"]))))
    tr_dataset = BaseDataset(conf, scp, phase="train", scaler=scaler)
    dev_dataset = BaseDataset(conf, scp, phase="dev", scaler=scaler)
    eval_dataset = BaseDataset(conf, scp, phase="eval", scaler=scaler)
    dataloader = {
        "spkrs": spkrs,
        "train": DataLoader(
            tr_dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=n_jobs
        ),
        "dev": DataLoader(
            dev_dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=n_jobs
        ),
        "eval": DataLoader(
            eval_dataset, batch_size=conf["batch_size"] * 10, num_workers=n_jobs
        ),
    }
    return dataloader
