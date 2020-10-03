#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. Kobayashi <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
VQVAE w/ LSGAN trainer

"""

import random
import torch
from crank.net.trainer.trainer_vqvae import VQVAETrainer


class LSGANTrainer(VQVAETrainer):
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
        self.gan_flag = False
        self.stop_generator = False
        self._check_gan_start()

    def check_custom_start(self):
        self._check_gan_start()

    def save_model(self):
        checkpoint = self.expdir / "checkpoint_{}steps.pkl".format(self.steps)
        state_dict = {
            "steps": self.steps,
            "model": {"G": self.model["G"].state_dict()},
        }
        if self.gan_flag:
            state_dict["model"]["D"] = self.model["D"].state_dict()
        torch.save(state_dict, checkpoint)

    def train(self, batch, phase="train"):
        loss = self._get_loss_dict()
        if self.gan_flag:
            loss = self.forward_lsgan(batch, loss, phase=phase)
        else:
            loss = self.forward_vqvae(batch, loss, phase=phase)
        loss_values = self._parse_loss(loss)
        self._flush_writer(loss, phase)
        return loss_values

    def forward_lsgan(self, batch, loss, phase="train"):
        if self.conf["train_first"] == "generator":
            loss = self.update_G(batch, loss, phase=phase)
            loss = self.update_D(batch, loss, phase=phase)
        else:
            loss = self.update_D(batch, loss, phase=phase)
            loss = self.update_G(batch, loss, phase=phase)
        loss["objective"] = loss["G"] + loss["D"]
        return loss

    def update_G(self, batch, loss, phase="train"):
        # train generator
        enc_h = self._generate_conditions(batch, encoder=True)
        dec_h = self._generate_conditions(batch)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]
        outputs = self.model["G"].forward(feats, enc_h=enc_h, dec_h=dec_h)
        loss = self.calculate_vqvae_loss(batch, outputs, loss)

        if self.conf["cvadv_flag"] and random.choice([True, False]):
            h_cv = self._generate_conditions(batch, use_cvfeats=True)
            cv_outputs = self.model["G"].forward(feats, enc_h=enc_h, dec_h=h_cv)
            decoded = cv_outputs["decoded"]
            h_scaler = batch["cv_h_scalar"]
        else:
            decoded = outputs["decoded"]
            h_scaler = batch["org_h_scalar"]
        loss = self.calculate_adv_loss(batch, decoded, h_scaler, loss)

        if phase == "train" and not self.stop_generator:
            self.optimizer["G"].zero_grad()
            loss["G"].backward()
            self.optimizer["G"].step()
        return loss

    def update_D(self, batch, loss, phase="train"):
        # train discriminator
        enc_h = self._generate_conditions(batch, encoder=True)
        dec_h = self._generate_conditions(batch)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]
        outputs = self.model["G"].forward(feats, enc_h=enc_h, dec_h=dec_h)
        if self.conf["cvadv_flag"] and random.choice([True, False]):
            dec_h_cv = self._generate_conditions(batch, use_cvfeats=True)
            cv_outputs = self.model["G"].forward(feats, enc_h=enc_h, dec_h=dec_h_cv)
            decoded = cv_outputs["decoded"]
            h_scaler = batch["cv_h_scalar"]
        else:
            decoded = outputs["decoded"]
            h_scaler = batch["org_h_scalar"]
        loss = self.calculate_discriminator_loss(batch, decoded, h_scaler, loss)

        if phase == "train":
            self.optimizer["D"].zero_grad()
            loss["D"].backward()
            self.optimizer["D"].step()
        return loss

    def calculate_adv_loss(self, batch, decoded, h_scaler, loss):
        mask = batch["mask"]
        outputs = self.model["D"].forward(decoded.transpose(1, 2)).transpose(1, 2)

        if self.conf["acgan_flag"]:
            outputs, spkr_cls = torch.split(outputs, [1, self.n_spkrs], dim=2)
            loss = self.calculate_acgan_loss(spkr_cls, batch["org_h_scalar"], loss)

        outputs = outputs.masked_select(mask)
        loss["adv"] = self.criterion["mse"](outputs, torch.ones_like(outputs))
        loss["G"] += self.conf["alphas"]["adv"] * loss["adv"]
        return loss

    def calculate_acgan_loss(
        self, spkr_cls, h_scalar, loss, label="adv", model="generator"
    ):
        loss["ce_{}".format(label)] = self.criterion["ce"](
            spkr_cls.reshape(-1, spkr_cls.size(2)), h_scalar.reshape(-1)
        )
        loss[model] += self.conf["alphas"]["ce"] * loss["ce_{}".format(label)]
        return loss

    def calculate_discriminator_loss(self, batch, decoded, h_scaler, loss):
        mask = batch["mask"]
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]
        fake_sample = (
            self.model["D"].forward(decoded.detach().transpose(1, 2)).transpose(1, 2)
        )
        real_sample = self.model["D"].forward(feats.transpose(1, 2)).transpose(1, 2)

        if self.conf["acgan_flag"]:
            fake_sample, spkr_cls_fake = torch.split(
                fake_sample, [1, self.n_spkrs], dim=2
            )
            real_sample, spkr_cls_real = torch.split(
                real_sample, [1, self.n_spkrs], dim=2
            )
            loss = self.calculate_acgan_loss(
                spkr_cls_fake, h_scaler, loss, label="fake", model="discriminator"
            )
            loss = self.calculate_acgan_loss(
                spkr_cls_real, h_scaler, loss, label="real", model="discriminator"
            )

        fake_sample = fake_sample.masked_select(mask)
        real_sample = real_sample.masked_select(mask)
        loss["fake"] = self.criterion["mse"](fake_sample, torch.zeros_like(fake_sample))
        loss["real"] = self.criterion["mse"](real_sample, torch.ones_like(real_sample))
        loss["D"] += (
            self.conf["alphas"]["fake"] * loss["fake"]
            + self.conf["alphas"]["real"] * loss["real"]
        )
        return loss

    def _check_gan_start(self):
        if self.steps > self.conf["n_steps_gan_start"]:
            self.gan_flag = True
            if self.conf["n_steps_stop_generator"] > 0:
                self.stop_generator = True
        if (
            self.steps
            > self.conf["n_steps_gan_start"] + self.conf["n_steps_stop_generator"]
        ):
            self.stop_generator = False
