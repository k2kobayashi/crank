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


def get_criterion(conf):
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
        ),
    }
    return criterion


def get_optimizer(net_conf, model):
    if net_conf["optimizer"] == "adam":
        Gopt = optim.Adam(model["G"].parameters(), lr=net_conf["lr"])
    elif net_conf["optimizer"] == "radam":
        Gopt = toptim.RAdam(model["G"].parameters(), lr=net_conf["lr"])
    elif net_conf["optimizer"] == "lamb":
        Gopt = Lamb(
            model["G"].parameters(),
            lr=net_conf["lr"],
            weight_decay=0.01,
            betas=(0.9, 0.999),
            adam=False,
        )
    optimizer = {"G": Gopt}

    if "D" in model:
        if net_conf["optimizer"] == "adam":
            Dopt = optim.Adam(model["D"].parameters(), lr=net_conf["discriminator_lr"])
        elif net_conf["optimizer"] == "radam":
            Dopt = toptim.RAdam(
                model["D"].parameters(), lr=net_conf["discriminator_lr"]
            )
        elif net_conf["optimizer"] == "lamb":
            Dopt = Lamb(
                model["D"].parameters(),
                lr=net_conf["lr"],
                weight_decay=0.01,
                betas=(0.9, 0.999),
                adam=False,
            )
        optimizer.update({"D": Dopt})

    if "SPKRADV" in model:
        if net_conf["spkradv_optimizer"] == "adam":
            SPKRADVopt = optim.Adam(
                model["SPKRADV"].parameters(), lr=net_conf["spkradv_lr"]
            )
        elif net_conf["spkradv_optimizer"] == "radam":
            SPKRADVopt = toptim.RAdam(
                model["SPKRADV"].parameters(), lr=net_conf["spkradv_lr"]
            )
        elif net_conf["spkradv_optimizer"] == "lamb":
            SPKRADVopt = Lamb(
                model["SPKRADV"].parameters(),
                lr=net_conf["spkradv_lr"],
                weight_decay=0.01,
                betas=(0.9, 0.999),
                adam=False,
            )
        elif net_conf["spkradv_optimizer"] == "sgd":
            SPKRADVopt = optim.SGD(
                model["SPKRADV"].parameters(),
                lr=net_conf["spkradv_lr"],
            )
        optimizer.update({"SPKRADV": SPKRADVopt})
    return optimizer


def get_scheduler(net_conf, optimizer):
    scheduler = {
        "G": StepLR(
            optimizer["G"],
            step_size=net_conf["lr_decay_step_size"],
            gamma=net_conf["lr_decay_size"],
        ),
    }
    if "D" in optimizer:
        scheduler.update(
            {
                "D": StepLR(
                    optimizer["D"],
                    step_size=net_conf["discriminator_lr_decay_step_size"],
                    gamma=net_conf["discriminator_lr_decay_size"],
                )
            }
        )
    if "SPKRADV" in optimizer:
        scheduler.update(
            {
                "SPKRADV": StepLR(
                    optimizer["SPKRADV"],
                    step_size=net_conf["spkradv_lr_decay_step_size"],
                    gamma=net_conf["spkradv_lr_decay_size"],
                )
            }
        )
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
        conf,
        scp,
        phase="train",
        scaler=scaler,
        batch_len=batch_len,
    )
    dev_dataset = BaseDataset(
        conf,
        scp,
        phase="dev",
        scaler=scaler,
        batch_len=batch_len,
    )
    eval_dataset = BaseDataset(
        conf,
        scp,
        phase="eval",
        scaler=scaler,
        batch_len=batch_len,
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
