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

from torch import nn, optim
import torch_optimizer as toptim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from pytorch_lamb import Lamb

from crank.net.trainer.dataset import calculate_maxflen, BaseDataset
from crank.net.module.loss import MultiSizeSTFTLoss


def get_criterion(conf):
    criterion = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "ce": nn.CrossEntropyLoss(ignore_index=-100),
        "kld": nn.KLDivLoss(reduction="mean"),
        "stft": MultiSizeSTFTLoss(**conf["stft_params"]),
    }
    return criterion


def get_optimizer(net_conf, model):
    if net_conf["optimizer"] == "adam":
        optimizer = {
            "generator": optim.Adam(model["G"].parameters(), lr=net_conf["lr"]),
            "discriminator": optim.Adam(
                model["D"].parameters(), lr=net_conf["discriminator_lr"]
            ),
        }
    elif net_conf["optimizer"] == "radam":
        optimizer = {
            "generator": toptim.RAdam(model["G"].parameters(), lr=net_conf["lr"]),
            "discriminator": toptim.RAdam(
                model["D"].parameters(), lr=net_conf["discriminator_lr"]
            ),
        }
    elif net_conf["optimizer"] == "lamb":
        optimizer = {
            "generator": Lamb(
                model["G"].parameters(),
                lr=net_conf["lr"],
                weight_decay=0.01,
                betas=(0.9, 0.999),
                adam=False,
            ),
            "discriminator": Lamb(
                model["D"].parameters(),
                lr=net_conf["lr"],
                weight_decay=0.01,
                betas=(0.9, 0.999),
                adam=False,
            ),
        }
    else:
        raise ValueError("optimizer must be [adam, radam, lamb]")
    return optimizer


def get_scheduler(net_conf, optimizer):
    scheduler = {
        "generator": StepLR(
            optimizer["generator"],
            step_size=net_conf["lr_decay_step_size"],
            gamma=net_conf["lr_decay_size"],
        ),
        "discriminator": StepLR(
            optimizer["discriminator"],
            step_size=net_conf["discriminator_lr_decay_step_size"],
            gamma=net_conf["discriminator_lr_decay_size"],
        ),
    }
    return scheduler


def get_dataloader(conf, scp, scaler, flag="train", n_jobs=10):

    if flag in ["train", "reconstruction"]:
        feats = list(scp["train"]["feats"].values()) + list(
            scp["dev"]["feats"].values()
        )
    elif flag in ["eval"]:
        feats = list(scp["eval"]["feats"].values())

    if flag in ["reconstruction", "eval"]:
        batch_len = calculate_maxflen(feats)
    elif conf["batch_len"] is not None:
        batch_len = conf["batch_len"]
    else:
        batch_len = calculate_maxflen(feats)

    spkrs = dict(zip(scp["train"]["spkrs"], range(len(scp["train"]["spkrs"]))))
    tr_dataset = BaseDataset(
        conf, scp, phase="train", scaler=scaler, batch_len=batch_len,
    )
    dev_dataset = BaseDataset(
        conf, scp, phase="dev", scaler=scaler, batch_len=batch_len,
    )
    eval_dataset = BaseDataset(
        conf, scp, phase="eval", scaler=scaler, batch_len=batch_len,
    )
    dataloader = {
        "spkrs": spkrs,
        "batch_len": batch_len,
        "train": DataLoader(
            tr_dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=n_jobs
        ),
        "dev": DataLoader(
            dev_dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=n_jobs
        ),
        "eval": DataLoader(
            eval_dataset, batch_size=conf["batch_size"] * 20, num_workers=n_jobs
        ),
    }
    return dataloader
