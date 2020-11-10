#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. Kobayashi <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
VQVAE trainer

"""

import random

import numpy as np
import torch
from crank.net.trainer import BaseTrainer
from crank.net.trainer.dataset import convert_f0, create_one_hot
from crank.utils import feat2hdf5, mlfb2wavf, to_numpy, world2wav
from joblib import Parallel, delayed
from torch.nn.utils import clip_grad_norm


class VQVAETrainer(BaseTrainer):
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
        super().__init__(
            model,
            optimizer,
            criterion,
            dataloader,
            writer,
            expdir,
            conf,
            feat_conf,
            scheduler=scheduler,
            scaler=scaler,
            resume=resume,
            device=device,
            n_jobs=n_jobs,
        )

    def train(self, batch, phase="train"):
        loss = self._get_loss_dict()
        loss = self.forward_vqvae(batch, loss, phase=phase)
        loss_values = self._parse_loss(loss)
        self._flush_writer(loss, phase)
        return loss_values

    @torch.no_grad()
    def dev(self, batch):
        loss_values = self.train(batch, phase="dev")
        for cv_spkr_name in random.sample(list(self.spkrs.keys()), self.n_cv_spkrs):
            enc_h = self._get_enc_h(batch)
            dec_h, spkrvec = self._get_dec_h(batch, cv_spkr_name=cv_spkr_name)
            outputs = self.model["G"](batch["feats"], enc_h, dec_h, spkrvec=spkrvec)
            self._generate_cvwav(
                batch,
                outputs,
                cv_spkr_name,
                tdir="dev_wav",
                save_hdf5=False,
                n_samples=self.n_dev_samples,
            )
        return loss_values

    @torch.no_grad()
    def reconstruction(self, batch, tdir="reconstruction"):
        enc_h = self._get_enc_h(batch)
        dec_h, spkrvec = self._get_dec_h(batch, cv_spkr_name=None)
        outputs = self.model["G"].forward(batch["feats"], enc_h, dec_h, spkrvec=spkrvec)
        self._generate_cvwav(batch, outputs, None, tdir=tdir, save_hdf5=True)

    @torch.no_grad()
    def eval(self, batch):
        for cv_spkr_name in self.spkrs.keys():
            enc_h = self._get_enc_h(batch)
            dec_h, spkrvec = self._get_dec_h(batch, cv_spkr_name=cv_spkr_name)
            outputs = self.model["G"](batch["feats"], enc_h, dec_h, spkrvec=spkrvec)
            self._generate_cvwav(
                batch, outputs, cv_spkr_name, tdir="eval_wav", save_hdf5=True
            )

    def forward_vqvae(self, batch, loss, phase="train"):
        # train generator
        enc_h = self._get_enc_h(batch)
        dec_h, spkrvec = self._get_dec_h(batch)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]
        outputs = self.model["G"].forward(feats, enc_h, dec_h, spkrvec=spkrvec)
        loss = self.calculate_vqvae_loss(batch, outputs, loss)

        if self.conf["use_spkradv_training"]:
            loss = self.calculate_spkradv_loss(batch, outputs, loss, phase=phase)

        # train classifier using converted feature
        if self.conf["train_cv_classifier"]:
            loss = self.calculate_cv_spkr_cls_loss(feats, batch, enc_h, loss)

        loss["objective"] += loss["G"]
        if phase == "train":
            self.step_model(loss, model="G")

        if phase == "train" and self.conf["use_spkradv_training"]:
            outputs = self.model["G"].forward(feats, enc_h, dec_h, spkrvec=spkrvec)
            loss = self.update_SPKRADV(batch, outputs, loss, phase=phase)
        return loss

    def step_model(self, loss, model="G"):
        self.optimizer[model].zero_grad()
        loss[model].backward()
        if self.conf["optim"][model]["clip_grad_norm"] != 0:
            clip_grad_norm(
                self.model[model].parameters(),
                self.conf["optim"][model]["clip_grad_norm"],
            )
        self.optimizer[model].step()

    def calculate_vqvae_loss(self, batch, outputs, loss):
        mask = batch["mask"]
        feats = batch["feats"]
        decoded = outputs["decoded"]
        loss["G_l1"] = self.criterion["fl1"](decoded, feats, mask=mask)
        loss["G_mse"] = self.criterion["fmse"](decoded, feats, mask=mask)
        loss["G_stft"] = self.criterion["fstft"](decoded, feats)
        if self.conf["encoder_spkr_classifier"]:
            loss["G_ce"] = self.criterion["ce"](
                outputs["spkr_cls"].reshape(-1, outputs["spkr_cls"].size(2)),
                batch["org_h_scalar"].reshape(-1),
            )

        # loss for vq
        encoded = outputs["encoded"]
        emb_idx = outputs["emb_idx"]
        for n in range(self.conf["n_vq_stacks"]):
            loss[f"G_commit{n}"] = self.criterion["mse"](
                encoded[n].masked_select(mask), emb_idx[n].masked_select(mask).detach()
            )
            if not self.conf["ema_flag"]:
                loss[f"G_dict{n}"] = self.criterion["mse"](
                    emb_idx[n].masked_select(mask),
                    encoded[n].masked_select(mask).detach(),
                )
        loss = self._parse_vqvae_loss(loss)
        return loss

    def calculate_spkradv_loss(self, batch, outputs, loss, phase="train"):
        advspkr_class = self.model["SPKRADV"].forward(outputs["encoded_unmod"])
        spkradv_loss = self.criterion["ce"](
            advspkr_class.reshape(-1, advspkr_class.size(2)),
            batch["org_h_scalar"].reshape(-1),
        )
        loss["G"] += self.conf["alpha"]["ce"] * spkradv_loss
        return loss

    def update_SPKRADV(self, batch, outputs, loss, phase="train"):
        advspkr_class = self.model["SPKRADV"].forward(
            outputs["encoded_unmod"], detach=True
        )
        spkradv_loss = self.criterion["ce"](
            advspkr_class.reshape(-1, advspkr_class.size(2)),
            batch["org_h_scalar"].reshape(-1),
        )
        loss["SPKRADV"] = self.conf["alpha"]["ce"] * spkradv_loss
        if phase == "train":
            self.step_model(loss, model="SPKRADV")
        return loss

    def _parse_vqvae_loss(self, loss):
        def _parse_vq(k):
            for n in range(self.conf["n_vq_stacks"]):
                loss["G"] += self.conf["alpha"][k] * loss[f"G_{k}{n}"]
            return loss

        for k in ["l1", "mse", "stft"]:
            loss["G"] += self.conf["alpha"][k] * loss[f"G_{k}"]
        if self.conf["encoder_spkr_classifier"]:
            loss["G"] += self.conf["alpha"]["ce"] * loss["G_ce"]
        loss = _parse_vq("commit")
        if not self.conf["ema_flag"]:
            loss = _parse_vq("dict")
        return loss

    def calculate_cv_spkr_cls_loss(self, feats, batch, enc_h, loss):
        dec_h_cv, spkrvec = self._get_dec_h(batch, use_cvfeats=True)
        cv_outputs = self.model["G"].forward(feats, enc_h, dec_h_cv, spkrvec=spkrvec)
        _, cv_spkr_cls = self.model["G"].encode(
            cv_outputs["decoded"].detach().transpose(1, 2)
        )

        loss["ce_cv"] = self.criterion["ce"](
            cv_spkr_cls.reshape(-1, cv_spkr_cls.size(2)),
            batch["cv_h_scalar"].reshape(-1),
        )
        loss["G"] += self.conf["alpha"]["ce"] * loss["ce_cv"]
        return loss

    def _get_enc_h(self, batch, use_cvfeats=False, cv_spkr_name=None):
        if self.conf["encoder_f0"]:
            f0, _, _ = self._prepare_feats(batch, cv_spkr_name, use_cvfeats)
            return f0
        else:
            return None

    def _get_dec_h(self, batch, use_cvfeats=False, cv_spkr_name=None):
        f0, h_onehot, h_scalar = self._prepare_feats(batch, cv_spkr_name, use_cvfeats)
        if not self.conf["use_spkr_embedding"]:
            if self.conf["decoder_f0"]:
                return torch.cat([f0, h_onehot], dim=-1), None
            else:
                return h_onehot, None
        else:
            if self.conf["decoder_f0"]:
                return f0, h_scalar
            else:
                return None, h_scalar

    def _prepare_feats(self, batch, cv_spkr_name, use_cvfeats=False):
        if cv_spkr_name is not None:
            # use specified cv speaker
            B, T, _ = batch["feats"].size()
            spkr_num = self.spkrs[cv_spkr_name]
            lcf0 = self._get_cvf0(batch, cv_spkr_name)
            h_onehot_np = create_one_hot(T, self.n_spkrs, spkr_num, B=B)
            h_onehot = torch.tensor(h_onehot_np).to(self.device)
            h_scalar = torch.ones((B, T)).long() * self.spkrs[cv_spkr_name]
            h_scalar = h_scalar.to(self.device)
        else:
            if use_cvfeats:
                # use randomly selected cv speaker by dataset
                lcf0 = batch["cv_lcf0"].clone()
                h_onehot = batch["cv_h_onehot"].clone()
                h_scalar = batch["cv_h_scalar"].clone()
            else:
                # use org speaker
                lcf0 = batch["lcf0"].clone()
                h_onehot = batch["org_h_onehot"].clone()
                h_scalar = batch["org_h_scalar"].clone()
        h_scalar[:, :] = h_scalar[:, 0:1]  # remove ignore_index (i.e., -100)
        return torch.cat([lcf0, batch["uv"]], axis=-1), h_onehot, h_scalar

    def _get_cvf0(self, batch, spkr_name):
        cv_lcf0s = []
        for n in range(batch["feats"].size(0)):
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
        if save_hdf5:
            self._save_decoded_to_hdf5(feats)
        if self.conf["feat_type"] == "mcep":
            self._save_decoded_world(feats, n_samples)
        else:
            self._save_decoded_mlfb(feats, n_samples)

    def _store_features(self, batch, outputs, cv_spkr_name, tdir):
        feats = {}
        feat_type = self.conf["feat_type"]
        for n in range(outputs["decoded"].size(0)):
            org_spkr_name = batch["org_spkr_name"][n]
            cv_name = org_spkr_name if cv_spkr_name is None else cv_spkr_name
            wavf = tdir / f"{batch['flbl'][n]}_org-{org_spkr_name}_cv-{cv_name}.wav"
            wavf.parent.mkdir(parents=True, exist_ok=True)

            # for feat
            feats[wavf] = {}
            flen = batch["flen"][n]
            feat = to_numpy(outputs["decoded"][n][:flen])
            if feat_type == "mcep" and not self.conf["use_mcep_0th"]:
                mcep_0th = to_numpy(batch["mcep_0th"][n][:flen])
                feat = np.hstack([mcep_0th, feat])
            feats[wavf]["feats"] = self.scaler[feat_type].inverse_transform(feat)
            feats[wavf]["normed_feat"] = feat

            # for f0 features
            org_cf0 = self.scaler["lcf0"].inverse_transform(
                to_numpy(batch["lcf0"][n][:flen])
            )
            cv_cf0 = convert_f0(self.scaler, org_cf0, org_spkr_name, cv_name)
            feats[wavf]["lcf0"] = cv_cf0
            feats[wavf]["normed_lcf0"] = self.scaler["lcf0"].transform(cv_cf0)
            feats[wavf]["uv"] = to_numpy(batch["uv"][n][:flen])
            feats[wavf]["f0"] = np.exp(cv_cf0) * feats[wavf]["uv"]

            if feat_type == "mcep":
                feats[wavf]["cap"] = to_numpy(batch["cap"][n][:flen])
        return feats

    def _save_decoded_to_hdf5(self, feats):
        type_features = ["feats", "normed_feat", "f0", "lcf0", "normed_lcf0", "uv"]
        if self.conf["feat_type"] == "mcep":
            type_features += ["cap"]
        for k in type_features:
            Parallel(n_jobs=self.n_jobs)(
                [
                    delayed(feat2hdf5)(feat[k], path, ext=k)
                    for path, feat in feats.items()
                ]
            )

    def _save_decoded_mlfb(self, feats, n_samples=-1):
        if n_samples == -1:
            n_samples = len(list(feats.keys()))
        if n_samples > len(list(feats.keys())):
            n_samples = len(list(feats.keys()))
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
                for wavf in random.sample(list(feats.keys()), n_samples)
            ]
        )

    def _save_decoded_world(self, feats, n_samples=-1):
        if n_samples == -1:
            n_samples = len(list(feats.keys()))
        if n_samples > len(list(feats.keys())):
            n_samples = len(list(feats.keys()))
        Parallel(n_jobs=self.n_jobs)(
            [
                delayed(world2wav)(
                    feats[k]["f0"][:, 0].astype(np.float64),
                    feats[k]["feats"].astype(np.float64),
                    feats[k]["cap"].astype(np.float64),
                    wavf=k,
                    fs=self.conf["feature"]["fs"],
                    fftl=self.conf["feature"]["fftl"],
                    shiftms=self.conf["feature"]["shiftms"],
                    alpha=self.conf["feature"]["mcep_alpha"],
                )
                for k in random.sample(list(feats.keys()), n_samples)
            ]
        )
