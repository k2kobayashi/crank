#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base trainer

"""

import logging
import torch
from abc import abstractmethod
from tqdm import tqdm
from pathlib import Path
from crank.utils import to_device


def TrainerWrapper(trainer_type, **ka):
    from crank.net.trainer import (
        VQVAETrainer,
        LSGANTrainer,
        CycleVQVAETrainer,
        CycleGANTrainer,
    )

    if trainer_type == "vqvae":
        trainer = VQVAETrainer(**ka)
    elif trainer_type == "lsgan":
        trainer = LSGANTrainer(**ka)
    elif trainer_type == "cycle":
        trainer = CycleVQVAETrainer(**ka)
    elif trainer_type == "cyclegan":
        trainer = CycleGANTrainer(**ka)
    else:
        raise NotImplementedError(
            "conf['trainer_type']: {} is not supported.".format(trainer_type)
        )
    return trainer


class BaseTrainer(object):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        dataloader,
        writer,
        expdir,
        conf,
        feat_conf,
        scheduler=None,
        scaler=None,
        resume=0,
        device="cuda",
        n_jobs=-1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.writer = writer
        self.expdir = Path(expdir)
        self.conf = conf
        self.feat_conf = feat_conf
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.n_jobs = n_jobs

        self.spkrs = dataloader["spkrs"]
        self.n_spkrs = len(self.spkrs)
        self.n_cv_spkrs = (
            self.conf["n_cv_spkrs"]
            if self.n_spkrs > self.conf["n_cv_spkrs"]
            else self.n_spkrs
        )

        self.resume_steps = resume
        self.steps = resume
        if not isinstance(self.scheduler, dict):
            self.scheduler.step(self.steps)
        else:
            self.scheduler["generator"].step(self.steps)
            self.scheduler["discriminator"].step(self.steps)

        self.finish_train = False
        self.tqdm = tqdm(initial=self.steps, total=self.conf["n_steps"], desc="train")

    @abstractmethod
    def train(self):
        loss_values = None
        return loss_values

    @abstractmethod
    def dev(self, batch):
        loss_values = None
        return loss_values

    @abstractmethod
    def eval(self, batch):
        pass

    @abstractmethod
    def reconstruction(self, batch):
        pass

    @abstractmethod
    def check_custom_start(self):
        pass

    def run(self, flag="train", tdir=None):
        self.flag = flag
        if flag == "train":
            while True:
                self._tr_step()
                if self.finish_train:
                    break
            self.tqdm.close()
            self.writer["train"].close()
            self.writer["dev"].close()
            logging.info("Finish training")
        else:
            self._run_eval(flag, tdir)

    def save_model(self):
        checkpoint = self.expdir / "checkpoint_{}steps.pkl".format(self.steps)
        state_dict = {
            "steps": self.steps,
            "model": {"G": self.model["G"].state_dict()},
        }
        torch.save(state_dict, checkpoint)

    def _run_eval(self, flag="eval", tdir=False):
        self.tqdm.close()
        if flag == "eval":
            logging.info("Run evaluation")
            self._eval_steps()
            logging.info("Finish evalation")
        if flag == "reconstruction":
            logging.info("Run reconstruction")
            self._reconstruction_steps(tdir)
            logging.info("Finish reconstruction")

    def _tr_step(self):
        for batch in self.dataloader["train"]:
            batch = to_device(batch, self.device)
            loss_values = self.train(batch, phase="train")
            if self.steps % self.conf["n_steps_print_loss"] == 0:
                self._print_loss_values(loss_values, phase="train")
            self._dev_step()

            # check step-by-step
            self._check_save_model()
            self._step_update()
            self._check_finish()

            # check custum func in each child
            self.check_custom_start()

    def _dev_step(self):
        if (
            self.steps % self.conf["dev_steps"] == 0
            and self.steps > self.conf["dev_steps"] - 1
            and self.steps != self.resume_steps
        ):
            dev_loss_values = self._get_loss_dict()
            for dev_idx, batch in enumerate(self.dataloader["dev"]):
                batch = to_device(batch, self.device)
                dev_loss_values = self.dev(batch)
                if dev_idx > 0:
                    break
            self._print_loss_values(dev_loss_values, phase="dev")

    def _eval_steps(self):
        eval_tqdm = tqdm(initial=0, total=len(self.dataloader["eval"]), desc="eval")
        for batch in self.dataloader["eval"]:
            batch = to_device(batch, self.device)
            self.eval(batch)
            eval_tqdm.update(1)
        eval_tqdm.close()

    def _reconstruction_steps(self, tdir=False):
        for dkey in ["train", "dev"]:
            recon_tqdm = tqdm(
                initial=0,
                total=len(self.dataloader[dkey]),
                desc="reconstruction ({})".format(dkey),
            )
            for batch in self.dataloader[dkey]:
                batch = to_device(batch, self.device)
                self.reconstruction(batch, tdir="reconstruction")
                recon_tqdm.update(1)
            recon_tqdm.close()

    def _get_loss_dict(self):
        loss_dict = {"objective": 0.0, "generator": 0.0, "discriminator": 0.0}
        return loss_dict

    def _parse_loss(self, loss):
        loss_values = self._get_loss_dict()
        for k in loss.keys():
            if k not in loss_values.keys():
                loss_values[k] = 0.0
            if isinstance(loss[k], torch.Tensor):
                loss_values[k] += loss[k].item()
        return loss_values

    def _print_loss_values(self, loss_values, phase="train"):
        print()
        logging.info("{} iterations: {}".format(phase, self.steps))
        for k, v in sorted(loss_values.items()):
            if v != 0.0:
                logging.info("{}: {}".format(k, v))

    def _flush_writer(self, loss, phase):
        if self.steps % self.conf["n_steps_print_loss"] == 0:
            for k in loss.keys():
                if isinstance(loss[k], torch.Tensor):
                    self.writer[phase].add_scalar(
                        "loss/{}".format(k), loss[k].item(), self.steps
                    )
            self.writer[phase].flush()

    def _check_save_model(self):
        if (self.resume_steps != self.steps) and (
            self.steps % self.conf["n_steps_save_model"] == 0
        ):
            self.save_model()

    def _step_update(self):
        self.steps += 1
        self.tqdm.update(1)
        if self.scheduler is not None:
            if not isinstance(self.scheduler, dict):
                self.scheduler.step()
            else:
                self.scheduler["generator"].step()
                self.scheduler["discriminator"].step()

    def _check_finish(self):
        if self.steps > self.conf["n_steps"]:
            self.finish_train = True
