#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. Kobayashi <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Cyclic VQVAE w/ StarGAN trainer

"""

import random

import torch
from crank.net.trainer import CycleVQVAETrainer, LSGANTrainer
from torch.nn.utils import clip_grad_norm


class StarGANTrainer(LSGANTrainer, CycleVQVAETrainer):
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
        if self.gan_flag:
            loss = self.forward_stargan(batch, loss, phase=phase)
        else:
            loss = self.forward_vqvae(batch, loss, phase=phase)
        loss_values = self._parse_loss(loss)
        self._flush_writer(loss, phase)
        return loss_values

    def forward_stargan(self, batch, loss, phase="train"):
        # run update_G and updata_D
        if self.conf["train_first"] == "G":
            loss = self.update_G(batch, loss, phase=phase)
            loss = self.update_D(batch, loss, phase=phase)
            loss = self.update_C(batch, loss, phase=phase)
        else:
            loss = self.update_D(batch, loss, phase=phase)
            loss = self.update_C(batch, loss, phase=phase)
            loss = self.update_G(batch, loss, phase=phase)
        loss["objective"] = loss["G"] + loss["D"] + loss["C"]
        return loss

    def update_G(self, batch, loss, phase="train"):
        # preapare aux. features
        enc_h = self._get_enc_h(batch)
        enc_h_cv = self._get_enc_h(batch, use_cvfeats=True)
        dec_h, spkrvec = self._get_dec_h(batch)
        dec_h_cv, spkrvec_cv = self._get_dec_h(batch, use_cvfeats=True)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]

        # VQVAE and cyclic vqvae loss
        cycle_outputs = self.model["G"].cycle_forward(
            feats, enc_h, dec_h, enc_h_cv, dec_h_cv, spkrvec, spkrvec_cv
        )
        vqvae_outputs = cycle_outputs[0]["org"]
        if self.conf["use_vqvae_loss"]:
            loss = self.calculate_vqvae_loss(batch, vqvae_outputs, loss)
        loss = self.calculate_cyclevqvae_loss(batch, cycle_outputs, loss)

        # SPKRADV loss
        if self.conf["use_spkradv_training"]:
            loss = self.calculate_spkradv_loss(batch, vqvae_outputs, loss, phase=phase)

        # StarGAN-based adversarial loss using cv one
        loss = self.calculate_starganadv_loss(batch, cycle_outputs, loss)

        # update G
        if phase == "train" and not self.stop_generator:
            self.optimizer["G"].zero_grad()
            loss["G"].backward()
            if self.conf["optim"]["G"]["clip_grad_norm"] != 0:
                clip_grad_norm(
                    self.model["G"].parameters(),
                    self.conf["optim"]["G"]["clip_grad_norm"],
                )
            self.optimizer["G"].step()

        # update SPKRADV
        if phase == "train" and self.conf["use_spkradv_training"]:
            outputs = self.model["G"].forward(feats, enc_h, dec_h, spkrvec=spkrvec)
            loss = self.update_SPKRADV(batch, outputs, loss, phase=phase)
        return loss

    def update_D(self, batch, loss, phase="train"):
        # preapare aux. features
        enc_h = self._get_enc_h(batch)
        enc_h_cv = self._get_enc_h(batch, use_cvfeats=True)
        dec_h, spkrvec = self._get_dec_h(batch)
        dec_h_cv, spkrvec_cv = self._get_dec_h(batch, use_cvfeats=True)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]

        # train discriminator
        outputs = self.model["G"].cycle_forward(
            feats, enc_h, dec_h, enc_h_cv, dec_h_cv, spkrvec, spkrvec_cv
        )
        loss = self.calculate_stargan_discriminator_loss(batch, outputs, loss)

        if phase == "train":
            self.optimizer["D"].zero_grad()
            loss["D"].backward()
            if self.conf["optim"]["D"]["clip_grad_norm"] != 0:
                clip_grad_norm(
                    self.model["D"].parameters(),
                    self.conf["optim"]["D"]["clip_grad_norm"],
                )
            self.optimizer["D"].step()
        return loss

    def update_C(self, batch, loss, phase="train"):
        def return_sample(x):
            return self.model["C"](x.transpose(1, 2)).transpose(1, 2)

        # get discriminator outputs
        real = return_sample(batch["feats"])
        real = real.reshape(-1, real.size(2))

        # calculate C loss
        h_scalar = batch["org_h_scalar"].reshape(-1)
        loss["C_real"] = self.criterion["ce"](real, h_scalar)
        loss["C"] += self.conf["alpha"]["acgan"] * loss["C_real"]
        if phase == "train":
            self.optimizer["C"].zero_grad()
            loss["C"].backward()
            if self.conf["optim"]["C"]["clip_grad_norm"] != 0:
                clip_grad_norm(
                    self.model["C"].parameters(),
                    self.conf["optim"]["C"]["clip_grad_norm"],
                )
            self.optimizer["C"].step()
        return loss

    def calculate_starganadv_loss(self, batch, outputs, loss):
        cv_decoded = outputs[0]["cv"]["decoded"]
        cv_inputs = self.get_D_inputs(batch, cv_decoded, label="cv")
        D_outputs = self.model["D"].forward(cv_inputs.transpose(1, 2)).transpose(1, 2)
        C_outputs = self.model["C"].forward(cv_decoded.transpose(1, 2)).transpose(1, 2)

        # calculate D loss
        valid_label = torch.ones_like(D_outputs)
        loss["D_adv_cv"] = self.criterion["mse"](D_outputs, valid_label)

        # calculate C loss
        h_scalar = batch["cv_h_scalar"].reshape(-1)
        C_outputs = C_outputs.reshape(-1, C_outputs.size(2))
        loss["C_adv_cv"] = self.criterion["ce"](C_outputs, h_scalar)

        # merge adv loss
        loss["G"] += (
            self.conf["alpha"]["adv"] * loss["D_adv_cv"]
            + self.conf["alpha"]["acgan"] * loss["C_adv_cv"]
        )
        return loss

    def calculate_stargan_discriminator_loss(self, batch, outputs, loss):
        def return_sample(x):
            return self.model["D"](x.transpose(1, 2)).transpose(1, 2)

        real_inputs = self.get_D_inputs(batch, batch["feats"], label="org")
        fake_inputs = self.get_D_inputs(
            batch, outputs[0]["cv"]["decoded"].detach(), label="cv"
        )

        # get discriminator outputs
        sample = {
            "real": return_sample(real_inputs),
            "cv_fake": return_sample(fake_inputs),
        }
        mask = batch["mask"]

        # loss by real
        real = sample["real"].masked_select(mask)
        loss["D_real"] = self.criterion["mse"](real, torch.ones_like(real))

        # loss by fake
        fake_key = random.choice(["cv_fake"])
        fake = sample[fake_key].masked_select(mask)
        loss["D_fake"] = self.criterion["mse"](fake, torch.zeros_like(fake))

        # merge D loss
        loss["D"] += (
            self.conf["alpha"]["fake"] * loss["D_fake"]
            + self.conf["alpha"]["real"] * loss["D_real"]
        )
        return loss
