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
from crank.net.trainer import BaseTrainer
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
        self.cycle_flag = False
        self._check_cycle_start()

    def check_custom_start(self):
        self._check_cycle_start()

    def train(self, batch, phase="train"):
        loss = self._get_loss_dict()
        if self.cycle_flag:
            loss = self.forward_cycle(batch, loss, phase=phase)
        else:
            loss = self.forward_vqvae(batch, loss, phase=phase)
        loss = self.forward_spkradv(batch, loss, phase=phase)
        loss = self.forward_spkrclassifier(batch, loss, phase=phase)
        loss_values = self._parse_loss(loss)
        self._flush_writer(loss, phase)
        return loss_values

    @torch.no_grad()
    def dev(self, batch):
        loss_values = self.train(batch, phase="dev")
        for cv_spkr_name in random.sample(list(self.spkrs.keys()), self.n_cv_spkrs):
            enc_h = self._get_enc_h(batch)
            dec_h, spkrvec = self._get_dec_h(batch, cv_spkr_name=cv_spkr_name)
            outputs = self.model["G"](batch["in_feats"], enc_h, dec_h, spkrvec=spkrvec)
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
        outputs = self.model["G"].forward(
            batch["in_feats"], enc_h, dec_h, spkrvec=spkrvec
        )
        self._generate_cvwav(batch, outputs, None, tdir=tdir, save_hdf5=True)

    @torch.no_grad()
    def eval(self, batch):
        for cv_spkr_name in self.spkrs.keys():
            enc_h = self._get_enc_h(batch)
            dec_h, spkrvec = self._get_dec_h(batch, cv_spkr_name=cv_spkr_name)
            outputs = self.model["G"](batch["in_feats"], enc_h, dec_h, spkrvec=spkrvec)
            self._generate_cvwav(
                batch, outputs, cv_spkr_name, tdir="eval_wav", save_hdf5=True
            )

    def forward_vqvae(self, batch, loss, phase="train"):
        enc_h = self._get_enc_h(batch)
        dec_h, spkrvec = self._get_dec_h(batch)
        feats = batch["in_feats"]
        outputs = self.model["G"].forward(feats, enc_h, dec_h, spkrvec=spkrvec)
        loss = self.calculate_vqvae_loss(batch, outputs, loss)

        if self.conf["use_spkradv_training"]:
            loss = self.calculate_spkradv_loss(batch, outputs, loss, phase=phase)

        loss["objective"] += loss["G"]
        if phase == "train":
            self.step_model(loss, model="G")

        return loss

    def forward_cycle(self, batch, loss, phase="train"):
        enc_h = self._get_enc_h(batch)
        enc_h_cv = self._get_enc_h(batch, use_cvfeats=True)
        dec_h, spkrvec = self._get_dec_h(batch)
        dec_h_cv, spkrvec_cv = self._get_dec_h(batch, use_cvfeats=True)
        feats = batch["in_feats"]
        cycle_outputs = self.model["G"].cycle_forward(
            feats, enc_h, dec_h, enc_h_cv, dec_h_cv, spkrvec, spkrvec_cv
        )
        vqvae_outputs = cycle_outputs[0]["org"]
        loss = self.calculate_vqvae_loss(batch, vqvae_outputs, loss)
        loss = self.calculate_cyclevqvae_loss(batch, cycle_outputs, loss)

        if self.conf["use_spkradv_training"]:
            loss = self.calculate_spkradv_loss(batch, vqvae_outputs, loss, phase=phase)

        loss["objective"] += loss["G"]
        if phase == "train":
            self.step_model(loss, model="G")
        return loss

    def forward_spkradv(self, batch, loss, phase="train"):
        if self.conf["use_spkradv_training"]:
            enc_h = self._get_enc_h(batch)
            dec_h, spkrvec = self._get_dec_h(batch)
            feats = batch["in_feats"]
            outputs = self.model["G"].forward(feats, enc_h, dec_h, spkrvec=spkrvec)
            advspkr_class = self.model["SPKRADV"].forward(
                outputs["encoded_unmod"], detach=True
            )
            spkradv_loss = self.criterion["ce"](
                advspkr_class.reshape(-1, advspkr_class.size(2)),
                batch["org_h"].reshape(-1),
            )
            loss["SPKRADV"] = self.conf["alpha"]["ce"] * spkradv_loss
            if phase == "train":
                self.step_model(loss, model="SPKRADV")
        return loss

    def forward_spkrclassifier(self, batch, loss, phase="train"):
        def return_sample(x):
            return self.model["C"](x.transpose(1, 2)).transpose(1, 2)

        if self.conf["use_spkr_classifier"]:
            real = return_sample(batch["in_feats"])
            real = real.reshape(-1, real.size(2))
            h = batch["org_h"].reshape(-1)
            loss["C_real"] = self.criterion["ce"](real, h)
            loss["C"] += self.conf["alpha"]["ce"] * loss["C_real"]
            if phase == "train":
                self.step_model(loss, model="C")
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
        target = batch["out_feats"]
        decoded = outputs["decoded"]
        loss["G_l1"] = self.criterion["fl1"](decoded, target, mask=mask)
        loss["G_mse"] = self.criterion["fmse"](decoded, target, mask=mask)
        loss["G_stft"] = self.criterion["fstft"](decoded, target)

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

    def calculate_cyclevqvae_loss(self, batch, outputs, loss):
        def calculate_spkrcls_loss(batch, outputs):
            def return_sample(x):
                return self.model["C"](x.transpose(1, 2)).transpose(1, 2)

            fake = return_sample(outputs["decoded"])
            fake = fake.reshape(-1, fake.size(2))
            h = batch["cv_h"].reshape(-1)
            return self.criterion["ce"](fake, h)

        mask = batch["mask"]
        for c in range(self.conf["n_cycles"]):
            for io in ["cv", "recon"]:
                lbl = f"{c}cyc_{io}"
                o = outputs[c][io]
                if io == "cv":
                    loss[f"C_fake_{lbl}"] = calculate_spkrcls_loss(batch, o)
                elif io == "recon":
                    target = batch["in_feats"]
                    decoded = o["decoded"]
                    loss[f"G_l1_{lbl}"] = self.criterion["fl1"](
                        decoded, target, mask=mask
                    )
                    loss[f"G_mse_{lbl}"] = self.criterion["fmse"](
                        decoded, target, mask=mask
                    )
                    loss[f"G_stft_{lbl}"] = self.criterion["fstft"](decoded, target)
                    for n in range(self.conf["n_vq_stacks"]):
                        loss[f"G_commit{n}_{lbl}"] = self.criterion["mse"](
                            o["encoded"][n].masked_select(mask),
                            o["emb_idx"][n].masked_select(mask).detach(),
                        )
                        if not self.conf["ema_flag"]:
                            loss[f"G_dict{n}_{lbl}"] = self.criterion["mse"](
                                o["emb_idx"][n].masked_select(mask),
                                o["encoded"][n].masked_select(mask).detach(),
                            )
        loss = self._parse_cyclevqvae_loss(loss)
        return loss

    def calculate_spkradv_loss(self, batch, outputs, loss, phase="train"):
        advspkr_class = self.model["SPKRADV"].forward(outputs["encoded_unmod"])
        loss["G_spkradv"] = self.criterion["ce"](
            advspkr_class.reshape(-1, advspkr_class.size(2)),
            batch["org_h"].reshape(-1),
        )
        loss["G"] += self.conf["alpha"]["ce"] * loss["G_spkradv"]
        return loss

    def _parse_vqvae_loss(self, loss):
        def _parse_vq(k):
            for n in range(self.conf["n_vq_stacks"]):
                loss["G"] += self.conf["alpha"][k] * loss[f"G_{k}{n}"]
            return loss

        for k in ["l1", "mse", "stft"]:
            loss["G"] += self.conf["alpha"][k] * loss[f"G_{k}"]
        loss = _parse_vq("commit")
        if not self.conf["ema_flag"]:
            loss = _parse_vq("dict")
        return loss

    def _parse_cyclevqvae_loss(self, loss):
        for c in range(self.conf["n_cycles"]):
            alpha_cycle = self.conf["alpha"]["cycle"]
            # for cv
            lbl = f"{c}cyc_cv"
            loss["G"] += alpha_cycle * self.conf["alpha"]["ce"] * loss[f"C_fake_{lbl}"]

            # for recon
            lbl = f"{c}cyc_recon"
            for k in ["l1", "mse", "stft"]:
                loss["G"] += alpha_cycle * self.conf["alpha"][k] * loss[f"G_{k}_{lbl}"]
            for n in range(self.conf["n_vq_stacks"]):
                loss["G"] += (
                    alpha_cycle
                    * self.conf["alpha"]["commit"]
                    * loss[f"G_commit{n}_{lbl}"]
                )
                if not self.conf["ema_flag"]:
                    loss["G"] += (
                        alpha_cycle
                        * self.conf["alpha"]["dict"]
                        * loss[f"G_dict{n}_{lbl}"]
                    )
        return loss

    def _check_cycle_start(self):
        if (
            self.conf["use_cyclic_training"]
            and self.steps > self.conf["n_steps_cycle_start"]
        ):
            self.cycle_flag = True

        if self.conf["use_cyclic_training"] and not self.conf["use_spkr_classifier"]:
            raise ValueError(
                "use_cyclic_training requires use_spkr_classifier to be true"
            )
