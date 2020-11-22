#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
BaseTrainer class

"""

import random
import logging
from abc import abstractmethod
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import torch
from tqdm import tqdm

from crank.utils import to_device
from crank.net.trainer.dataset import convert_f0, create_one_hot
from crank.utils import feat2hdf5, mlfb2wavf, to_numpy, world2wav


def TrainerWrapper(trainer_type, **ka):
    from crank.net.trainer import (
        CycleGANTrainer,
        LSGANTrainer,
        VQVAETrainer,
        StarGANTrainer,
    )

    if trainer_type == "vqvae":
        trainer = VQVAETrainer(**ka)
    elif trainer_type == "lsgan":
        trainer = LSGANTrainer(**ka)
    elif trainer_type == "cyclegan":
        trainer = CycleGANTrainer(**ka)
    elif trainer_type == "stargan":
        trainer = StarGANTrainer(**ka)
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
        self.n_cv_spkrs = 4 if self.n_spkrs > 4 else self.n_cv_spkrs
        self.n_dev_samples = 5

        self.resume_steps = resume
        self.steps = resume
        if not isinstance(self.scheduler, dict):
            self.scheduler.step(self.steps)
        else:
            for k in self.scheduler:
                self.scheduler[k].step(self.steps)

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
        for m in ["SPKRADV", "D", "C"]:
            if m in self.model.keys():
                state_dict["model"].update({m: self.model[m].state_dict()})
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
        loss_dict = {"objective": 0.0, "G": 0.0, "D": 0.0, "C": 0.0, "SPKRADV": 0.0}
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
                for k in self.scheduler:
                    self.scheduler[k].step(self.steps)

    def _check_finish(self):
        if self.steps > self.conf["n_steps"]:
            self.finish_train = True

    def _get_enc_h(self, batch, use_cvfeats=False, cv_spkr_name=None):
        if self.conf["encoder_f0"]:
            f0, _, _ = self._prepare_conditions(batch, cv_spkr_name, use_cvfeats)
            return f0
        else:
            return None

    def _get_dec_h(self, batch, use_cvfeats=False, cv_spkr_name=None):
        f0, h, h_onehot = self._prepare_conditions(batch, cv_spkr_name, use_cvfeats)
        if not self.conf["use_spkr_embedding"]:
            if self.conf["decoder_f0"]:
                return torch.cat([f0, h_onehot], dim=-1), None
            else:
                return h_onehot, None
        else:
            if self.conf["decoder_f0"]:
                return f0, h
            else:
                return None, h

    def _prepare_conditions(self, batch, cv_spkr_name, use_cvfeats=False):
        if cv_spkr_name is not None:
            # use specified cv speaker
            B, T, _ = batch["in_feats"].size()
            spkr_num = self.spkrs[cv_spkr_name]
            lcf0 = self._get_cvf0(batch, cv_spkr_name)
            h_onehot_np = create_one_hot(T, self.n_spkrs, spkr_num, B=B)
            h_onehot = torch.tensor(h_onehot_np).to(self.device)
            h = (torch.ones((B, T)).long() * self.spkrs[cv_spkr_name]).to(self.device)
        else:
            if use_cvfeats:
                # use randomly selected cv speaker by dataset
                lcf0 = batch["cv_lcf0"]
                h = batch["cv_h"].clone()
                h_onehot = batch["cv_h_onehot"]
            else:
                # use org speaker
                lcf0 = batch["lcf0"]
                h_onehot = batch["org_h_onehot"]
                h = batch["org_h"].clone()
        h[:, :] = h[:, 0:1]  # remove ignore_index (i.e., -100)
        f0 = torch.cat([lcf0, batch["uv"]], axis=-1)
        return f0, h, h_onehot

    def _get_cvf0(self, batch, spkr_name):
        cv_lcf0s = []
        for n in range(batch["in_feats"].size(0)):
            org_lcf0 = self.scaler["lcf0"].inverse_transform(to_numpy(batch["lcf0"][n]))
            cv_lcf0 = convert_f0(
                self.scaler, org_lcf0, batch["org_spkr_name"][n], spkr_name
            )
            normed_cv_lcf0 = self.scaler["lcf0"].transform(cv_lcf0)
            cv_lcf0s.append(torch.tensor(normed_cv_lcf0))
        return torch.stack(cv_lcf0s, dim=0).float().to(self.device)

    def _generate_cvwav(
        self,
        batch,
        outputs,
        cv_spkr_name=None,
        tdir="dev_wav",
        save_hdf5=True,
        n_samples=1,
    ):
        tdir = self.expdir / tdir / str(self.steps)
        feats = self._store_features(batch, outputs, cv_spkr_name, tdir)
        if not (n_samples == -1 or n_samples > len(feats.keys())):
            feats = dict((k, feats[k]) for k in random.sample(feats.keys(), n_samples))
        for k in feats.keys():
            Path(k).parent.mkdir(parents=True, exist_ok=True)
        if save_hdf5:
            self._save_decoded_to_hdf5(feats)
        if self.conf["output_feat_type"] == "mcep":
            self._save_decoded_world(feats)
        else:
            self._save_decoded_mlfb(feats)

    def _store_features(self, batch, outputs, cv_spkr_name, tdir):
        def inv_trans(k, feat):
            if k not in self.conf["ignore_scaler"]:
                return self.scaler[k].inverse_transform(feat)
            else:
                return feat

        feats = {}
        feat_type = self.conf["output_feat_type"]
        for n in range(outputs["decoded"].size(0)):
            org_spkr_name = batch["org_spkr_name"][n]
            cv_name = org_spkr_name if cv_spkr_name is None else cv_spkr_name
            wavf = tdir / f"{batch['flbl'][n]}_org-{org_spkr_name}_cv-{cv_name}.wav"

            # feat
            feats[wavf] = {}
            flen = batch["flen"][n]
            feat = to_numpy(outputs["decoded"][n][:flen])
            if feat_type == "mcep":
                feats[wavf]["cap"] = to_numpy(batch["cap"][n][:flen])
                if not self.conf["use_mcep_0th"]:
                    org_mcep_0th = to_numpy(batch["mcep_0th"][n][:flen])
                    org_mcep = to_numpy(batch["in_feats"][n][:flen])
                    feat = np.ascontiguousarray(np.hstack([org_mcep_0th, feat]))
                    rmcep = np.ascontiguousarray(np.hstack([org_mcep_0th, org_mcep]))
                    feats[wavf]["rmcep"] = inv_trans(feat_type, rmcep)
                else:
                    feats[wavf]["rmcep"] = None
            feats[wavf]["feats"] = inv_trans(feat_type, feat)

            # f0
            org_cf0 = inv_trans("lcf0", to_numpy(batch["lcf0"][n][:flen]))
            cv_cf0 = convert_f0(self.scaler, org_cf0, org_spkr_name, cv_name)
            feats[wavf]["lcf0"] = cv_cf0
            feats[wavf]["uv"] = to_numpy(batch["uv"][n][:flen])
            feats[wavf]["f0"] = np.exp(cv_cf0) * feats[wavf]["uv"]

            # save normed one as well
            feats[wavf]["normed_lcf0"] = self.scaler["lcf0"].transform(cv_cf0)
            feats[wavf]["normed_feat"] = feat
        return feats

    def _save_decoded_to_hdf5(self, feats):
        type_features = ["feats", "normed_feat", "f0", "lcf0", "normed_lcf0", "uv"]
        if self.conf["output_feat_type"] == "mcep":
            type_features += ["cap"]
        for k in type_features:
            Parallel(n_jobs=self.n_jobs)(
                [
                    delayed(feat2hdf5)(feat[k], path, ext=k)
                    for path, feat in feats.items()
                ]
            )

    def _save_decoded_mlfb(self, feats):
        Parallel(n_jobs=self.n_jobs)(
            [
                delayed(mlfb2wavf)(
                    feats[wavf]["feats"],
                    wavf,
                    fs=self.feat_conf["fs"],
                    n_mels=self.feat_conf["mlfb_dim"],
                    fftl=self.feat_conf["fftl"],
                    hop_size=self.feat_conf["hop_size"],
                    fmin=self.feat_conf["fmin"],
                    fmax=self.feat_conf["fmax"],
                    plot=True,
                )
                for wavf in feats.keys()
            ]
        )

    def _save_decoded_world(self, feats):
        Parallel(n_jobs=self.n_jobs)(
            [
                delayed(world2wav)(
                    feats[k]["f0"][:, 0].astype(np.float64),
                    feats[k]["feats"].astype(np.float64),
                    feats[k]["cap"].astype(np.float64),
                    rmcep=feats[k]["rmcep"].astype(np.float64),
                    wavf=k,
                    fs=self.conf["feature"]["fs"],
                    fftl=self.conf["feature"]["fftl"],
                    shiftms=self.conf["feature"]["shiftms"],
                    alpha=self.conf["feature"]["mcep_alpha"],
                )
                for k in feats.keys()
            ]
        )
