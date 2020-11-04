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

import torch
from crank.net.trainer.trainer_vqvae import VQVAETrainer
from torch.nn.utils import clip_grad_norm


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
        if self.conf["train_first"] == "G":
            loss = self.update_G(batch, loss, phase=phase)
            loss = self.update_D(batch, loss, phase=phase)
        else:
            loss = self.update_D(batch, loss, phase=phase)
            loss = self.update_G(batch, loss, phase=phase)
        loss["objective"] = loss["G"] + loss["D"]
        return loss

    def update_G(self, batch, loss, phase="train"):
        # train generator
        enc_h = self._get_enc_h(batch)
        dec_h, spkrvec = self._get_dec_h(batch)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]
        outputs = self.model["G"].forward(feats, enc_h, dec_h, spkrvec)
        loss = self.calculate_vqvae_loss(batch, outputs, loss)
        if self.conf["use_spkradv_training"]:
            loss = self.calculate_spkradv_loss(batch, outputs, loss, phase=phase)

        if self.conf["cvadv_flag"]:
            dec_h, spkrvec = self._get_dec_h(batch, use_cvfeats=True)
            h_scalar = batch["cv_h_scalar"]
        else:
            h_scalar = batch["org_h_scalar"]
        adv_outputs = self.model["G"].forward(
            feats,
            enc_h,
            dec_h,
            spkrvec=spkrvec,
            use_ema=False,
            encoder_detach=self.conf["encoder_detach"],
        )
        adv_decoded = adv_outputs["decoded"]
        loss = self.calculate_adv_loss(
            batch, adv_decoded, h_scalar, batch["mask"], loss
        )

        if phase == "train" and not self.stop_generator:
            self.optimizer["G"].zero_grad()
            loss["G"].backward()
            if self.conf["optim"]["G"]["clip_grad_norm"] != 0:
                clip_grad_norm(
                    self.model["G"].parameters(),
                    self.conf["optim"]["G"]["clip_grad_norm"],
                )
            self.optimizer["G"].step()

        if phase == "train" and self.conf["use_spkradv_training"]:
            outputs = self.model["G"].forward(feats, enc_h, dec_h, spkrvec=spkrvec)
            loss = self.update_SPKRADV(batch, outputs, loss, phase=phase)
        return loss

    def update_D(self, batch, loss, phase="train"):
        def return_sample(x):
            return self.model["D"](x.transpose(1, 2)).transpose(1, 2)

        enc_h = self._get_enc_h(batch)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]
        mask = batch["mask"]

        # calculate fake D loss
        if self.conf["cvadv_flag"]:
            dec_h, spkrvec = self._get_dec_h(batch, use_cvfeats=True)
            h_scalar = batch["cv_h_scalar"]
        else:
            dec_h, spkrvec = self._get_dec_h(batch)
            h_scalar = batch["org_h_scalar"]
        outputs = self.model["G"].forward(feats, enc_h, dec_h, spkrvec)
        real_inputs = self.get_D_inputs(batch, batch["feats"], label="org")
        fake_inputs = self.get_D_inputs(batch, outputs["decoded"].detach(), label="cv")

        # get discriminator outputs
        sample = {
            "real": return_sample(real_inputs),
            "cv_fake": return_sample(fake_inputs),
        }
        loss = self.calculate_discriminator_loss(
            sample["cv_fake"], h_scalar, mask, loss, label="fake"
        )
        loss = self.calculate_discriminator_loss(
            sample["real"], batch["org_h_scalar"], mask, loss, label="real"
        )

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

    def calculate_adv_loss(self, batch, decoded, h_scalar, mask, loss):
        fake_inputs = self.get_D_inputs(batch, decoded, label="cv")
        fake = self.model["D"].forward(fake_inputs.transpose(1, 2)).transpose(1, 2)

        if self.conf["acgan_flag"]:
            fake, spkr_cls = torch.split(fake, [1, self.n_spkrs], dim=2)
            loss = self.calculate_acgan_loss(spkr_cls, h_scalar, loss)

        fake = fake.masked_select(mask)
        loss["D_adv"] = self.criterion["mse"](fake, torch.ones_like(fake))
        loss["G"] += self.conf["alpha"]["adv"] * loss["D_adv"]
        return loss

    def calculate_discriminator_loss(self, sample, h_scalar, mask, loss, label="real"):
        if self.conf["acgan_flag"]:
            sample, spkr_cls = torch.split(sample, [1, self.n_spkrs], dim=2)
            loss = self.calculate_acgan_loss(
                spkr_cls, h_scalar, loss, label=label, model="D"
            )
        sample = sample.masked_select(mask)
        if label == "real":
            correct_label = torch.ones_like(sample)
        else:
            correct_label = torch.zeros_like(sample)
        loss[f"D_{label}"] = self.criterion["mse"](sample, correct_label)
        loss["D"] += self.conf["alpha"][label] * loss[f"D_{label}"]
        return loss

    def calculate_acgan_loss(self, spkr_cls, h_scalar, loss, label="adv", model="G"):
        loss[f"D_acgan_{label}"] = self.criterion["ce"](
            spkr_cls.reshape(-1, spkr_cls.size(2)), h_scalar.reshape(-1)
        )
        if not (self.conf["use_real_only_acgan"] and label == "fake"):
            loss[model] += self.conf["alpha"]["acgan"] * loss[f"D_acgan_{label}"]
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

    def get_D_inputs(self, batch, feats, label="org"):
        feats_list = [feats]
        if self.conf["use_D_uv"]:
            feats_list.append(batch["uv"])
        if self.conf["use_D_spkrcode"]:
            if not self.conf["use_spkr_embedding"]:
                feats_list.append(batch[f"{label}_h_onehot"])
            else:
                # remove ignore_index (i.e., -100)
                h_scalar = batch[f"{label}_h_scalar"].clone()
                h_scalar[:, :] = h_scalar[:, 0:1]
                feats_list.append(self.model["G"].spkr_embedding(h_scalar).detach())
        return torch.cat(feats_list, axis=-1).float()
