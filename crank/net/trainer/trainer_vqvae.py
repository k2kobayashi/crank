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
import torch
import numpy as np
from joblib import Parallel, delayed

from crank.utils import mlfb2wavf, feat2hdf5, world2wav, to_numpy
from crank.net.trainer import BaseTrainer
from crank.net.trainer.dataset import create_one_hot, convert_f0


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
            enc_h = self._generate_conditions(batch, encoder=True)
            dec_h = self._generate_conditions(batch, cv_spkr_name=cv_spkr_name)
            outputs = self.model["G"].forward(batch["feats"], enc_h=enc_h, dec_h=dec_h)
            self._generate_cvwav(batch, outputs, cv_spkr_name, tdir="dev_wav")
        return loss_values

    @torch.no_grad()
    def reconstruction(self, batch, tdir="reconstruction"):
        self.conf["n_gl_samples"] = 1
        enc_h = self._generate_conditions(batch, encoder=True)
        dec_h = self._generate_conditions(batch, cv_spkr_name=None)
        outputs = self.model["G"].forward(batch["feats"], enc_h=enc_h, dec_h=dec_h)
        self._generate_cvwav(batch, outputs, None, tdir=tdir)

        if self.conf["cycle_reconstruction"]:
            recondir = self.expdir / tdir / str(self.steps)
            for cv_spkr_name in self.spkrs.keys():
                enc_h_cv = self._generate_conditions(
                    batch, cv_spkr_name=cv_spkr_name, encoder=True
                )
                dec_h_cv = self._generate_conditions(batch, cv_spkr_name=cv_spkr_name)
                cycle_outputs = self.model["G"].cycle_forward(
                    batch["feats"],
                    org_enc_h=enc_h,
                    org_dec_h=dec_h,
                    cv_enc_h=enc_h_cv,
                    cv_dec_h=dec_h_cv,
                )
                recon = cycle_outputs[0]["recon"]["decoded"]

                for n in range(recon.size(0)):
                    org_spkr_name = batch["org_spkr_name"][n]
                    cv_name = org_spkr_name if cv_spkr_name is None else cv_spkr_name
                    wavf = recondir / "{}_org-{}_cv-{}.wav".format(
                        batch["flbl"][n], org_spkr_name, org_spkr_name
                    )
                    flen = batch["flen"][n]
                    normed_feat = to_numpy(recon[n][:flen])
                    feat = self.scaler[self.conf["feat_type"]].inverse_transform(
                        normed_feat
                    )
                    feat2hdf5(
                        feat,
                        wavf,
                        ext="feats_recon_{}-{}-{}".format(
                            org_spkr_name, cv_name, org_spkr_name
                        ),
                    )

    @torch.no_grad()
    def eval(self, batch):
        self.conf["n_gl_samples"] = 1
        for cv_spkr_name in self.spkrs.keys():
            enc_h = self._generate_conditions(batch, encoder=True)
            dec_h = self._generate_conditions(batch, cv_spkr_name=cv_spkr_name)
            outputs = self.model["G"].forward(batch["feats"], enc_h=enc_h, dec_h=dec_h)
            self._generate_cvwav(batch, outputs, cv_spkr_name, tdir="eval_wav")

    def forward_vqvae(self, batch, loss, phase="train"):
        # train generator
        enc_h = self._generate_conditions(batch, encoder=True)
        dec_h = self._generate_conditions(batch)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]
        outputs = self.model["G"].forward(feats, enc_h=enc_h, dec_h=dec_h)
        loss = self.calculate_vqvae_loss(batch, outputs, loss)

        # Train clasifier using converted feature
        if self.conf["train_cv_classifier"]:
            dec_h_cv = self._generate_conditions(batch, use_cvfeats=True)
            cv_outputs = self.model["G"].forward(feats, enc_h=enc_h, dec_h=dec_h_cv)
            _, cv_spkr_cls = self.model["G"].encode(
                cv_outputs["decoded"].detach().transpose(1, 2)
            )
            loss = self.calculate_cv_spkr_cls_loss(
                batch, cv_spkr_cls.transpose(1, 2), loss
            )

        loss["objective"] += loss["generator"]
        if phase == "train":
            self.optimizer["generator"].zero_grad()
            loss["generator"].backward()
            self.optimizer["generator"].step()
        return loss

    def calculate_vqvae_loss(self, batch, outputs, loss):
        mask = batch["mask"]

        # loss for reconstruction
        decoded = outputs["decoded"].masked_select(mask)
        spkr_cls = outputs["spkr_cls"]
        loss["l1"] = self.criterion["l1"](batch["feats"].masked_select(mask), decoded)
        loss["mse"] = self.criterion["mse"](batch["feats"].masked_select(mask), decoded)
        loss["stft"] = self.criterion["stft"](batch["feats"], outputs["decoded"])
        loss["ce"] = self.criterion["ce"](
            spkr_cls.reshape(-1, spkr_cls.size(2)), batch["org_h_scalar"].reshape(-1)
        )

        # loss for vq
        encoded = outputs["encoded"]
        emb_idx = outputs["emb_idx"]
        for n in range(self.conf["n_vq_stacks"]):
            loss["commit{}".format(n)] = self.criterion["mse"](
                encoded[n].masked_select(mask), emb_idx[n].masked_select(mask).detach()
            )
            if not self.conf["ema_flag"]:
                loss["dict{}".format(n)] = self.criterion["mse"](
                    emb_idx[n].masked_select(mask),
                    encoded[n].masked_select(mask).detach(),
                )
        loss = self._parse_vqvae_loss(loss)
        return loss

    def _parse_vqvae_loss(self, loss):
        def _parse_vq(k):
            for n in range(self.conf["n_vq_stacks"]):
                loss["generator"] += (
                    self.conf["alphas"][k][n] * loss["{}{}".format(k, n)]
                )
            return loss

        for k in ["l1", "mse", "ce", "stft"]:
            loss["generator"] += self.conf["alphas"][k] * loss[k]
        loss = _parse_vq("commit")
        if not self.conf["ema_flag"]:
            loss = _parse_vq("dict")
        return loss

    def calculate_cv_spkr_cls_loss(self, batch, cv_spkr_cls, loss):
        loss["ce_cv"] = self.criterion["ce"](
            cv_spkr_cls.reshape(-1, cv_spkr_cls.size(2)),
            batch["cv_h_scalar"].reshape(-1),
        )
        loss["generator"] += self.conf["alphas"]["ce"] * loss["ce_cv"]
        return loss

    def _generate_conditions(
        self, batch, cv_spkr_name=None, use_cvfeats=False, encoder=False
    ):
        # create lcf0, uv, h_onehot
        if cv_spkr_name is not None:
            spkr_num = self.spkrs[cv_spkr_name]
            B, T, _ = batch["feats"].size()
            lcf0 = torch.tensor(self._get_cvf0(batch, cv_spkr_name)).to(self.device)
            uv = batch["uv"]
            h_onehot = torch.tensor(create_one_hot(T, self.n_spkrs, spkr_num, B=B)).to(
                self.device
            )
        else:
            if use_cvfeats:
                lcf0, uv, h_onehot = batch["cv_lcf0"], batch["uv"], batch["cv_h_onehot"]
            else:
                lcf0, uv, h_onehot = batch["lcf0"], batch["uv"], batch["org_h_onehot"]

        # return conditions
        if encoder:
            if self.conf["encoder_f0"]:
                return torch.cat([lcf0, uv], dim=-1).to(self.device)
            else:
                return None
        else:
            if self.conf["decoder_f0"]:
                return torch.cat([lcf0, uv, h_onehot], dim=-1).to(self.device)
            else:
                return h_onehot.to(self.device)

    def _get_cvf0(self, batch, spkr_name):
        cv_lcf0s = []
        for n in range(batch["feats"].size(0)):
            cv_lcf0 = self.scaler["lcf0"].transform(
                convert_f0(
                    self.scaler,
                    self.scaler["lcf0"].inverse_transform(to_numpy(batch["lcf0"][n])),
                    batch["org_spkr_name"][n],
                    spkr_name,
                )
            )
            cv_lcf0s.append(torch.tensor(cv_lcf0))
        return torch.stack(cv_lcf0s, dim=0).float()

    def _generate_cvwav(self, batch, outputs, cv_spkr_name=None, tdir="dev_wav"):
        tdir = self.expdir / tdir / str(self.steps)
        feats = self._store_features(batch, outputs, cv_spkr_name, tdir)

        # generate wav
        if self.conf["feat_type"] == "mcep":
            self._save_decoded_world(feats)
        else:
            self._save_decoded_mlfbs(feats)

    def _store_features(self, batch, outputs, cv_spkr_name, tdir):
        feats = {}
        feat_type = self.conf["feat_type"]
        decoded = outputs["decoded"]
        for n in range(decoded.size(0)):
            org_spkr_name = batch["org_spkr_name"][n]
            cv_name = org_spkr_name if cv_spkr_name is None else cv_spkr_name
            wavf = tdir / "{}_org-{}_cv-{}.wav".format(
                batch["flbl"][n], org_spkr_name, cv_name
            )
            wavf.parent.mkdir(parents=True, exist_ok=True)

            # for feat
            flen = batch["flen"][n]
            feat = to_numpy(decoded[n][:flen])
            feats[wavf] = {}
            feats[wavf]["feat"] = self.scaler[feat_type].inverse_transform(feat)
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

    def _save_decoded_world(self, feats):
        for k, v in feats.items():
            world2wav(
                v["f0"][:, 0].astype(np.float64),
                v["feat"].astype(np.float64),
                v["cap"].astype(np.float64),
                wavf=k,
                fs=self.conf["feature"]["fs"],
                fftl=self.conf["feature"]["fftl"],
                shiftms=self.conf["feature"]["shiftms"],
                alpha=self.conf["feature"]["mcep_alpha"],
            )

    def _save_decoded_mlfbs(self, feats):
        n_samples = (
            self.conf["n_gl_samples"]
            if len(feats.keys()) > self.conf["n_gl_samples"]
            else len(feats.keys())
        )

        # gl
        Parallel(n_jobs=self.n_jobs)(
            [
                delayed(mlfb2wavf)(
                    feats[wavf]["feat"],
                    wavf,
                    fs=self.feat_conf["fs"],
                    n_mels=self.feat_conf["mlfb_dim"],
                    fftl=self.feat_conf["fftl"],
                    hop_size=self.feat_conf["hop_size"],
                    plot=True,
                )
                for wavf in random.sample(list(feats.keys()), n_samples)
            ]
        )

        # save as hdf5
        k = "normed_feat" if self.conf["save_mlfb_type"] == "normed" else "feat"
        Parallel(n_jobs=self.n_jobs)(
            [
                delayed(feat2hdf5)(feat[k], path, ext="feats")
                for path, feat in feats.items()
            ]
        )

        if self.conf["save_f0_feats"]:
            type_features = ["lcf0", "f0", "normed_lcf0", "uv"]
            if self.conf["feat_type"] == "mcep":
                type_features += ["cap"]
            for k in type_features:
                Parallel(n_jobs=self.n_jobs)(
                    [
                        delayed(feat2hdf5)(feat[k], path, ext=k)
                        for path, feat in feats.items()
                    ]
                )
