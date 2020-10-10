#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. Kobayashi <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Cyclic VQVAE w/ LSGAN trainer

"""

import random
import torch
from crank.net.trainer import LSGANTrainer, CycleVQVAETrainer


class CycleGANTrainer(LSGANTrainer, CycleVQVAETrainer):
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
            loss = self.forward_cyclegan(batch, loss, phase=phase)
        else:
            loss = self.forward_vqvae(batch, loss, phase=phase)
        loss_values = self._parse_loss(loss)
        self._flush_writer(loss, phase)
        return loss_values

    def forward_cyclegan(self, batch, loss, phase="train"):
        return self.forward_lsgan(batch, loss, phase=phase)

    def update_G(self, batch, loss, phase="train"):
        enc_h = self._generate_conditions(batch, encoder=True)
        enc_h_cv = self._generate_conditions(batch, use_cvfeats=True, encoder=True)
        dec_h = self._generate_conditions(batch)
        dec_h_cv = self._generate_conditions(batch, use_cvfeats=True)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]

        # cycle loss
        cycle_outputs = self.model["G"].cycle_forward(
            feats,
            org_enc_h=enc_h,
            org_dec_h=dec_h,
            cv_enc_h=enc_h_cv,
            cv_dec_h=dec_h_cv,
        )
        loss = self.calculate_vqvae_loss(batch, cycle_outputs[0]["org"], loss)
        loss = self.calculate_cyclevqvae_loss(batch, cycle_outputs, loss)

        if self.conf["speaker_adversarial"]:
            loss = self.calculate_spkradv_loss(
                batch, cycle_outputs[0]["org"], loss, phase=phase
            )

        # adversarial loss for org and cv
        loss = self.calculate_cycleadv_loss(batch, cycle_outputs, loss)

        if phase == "train" and not self.stop_generator:
            self.optimizer["G"].zero_grad()
            loss["G"].backward()
            self.optimizer["G"].step()
        return loss

    def update_D(self, batch, loss, phase="train"):
        enc_h = self._generate_conditions(batch, encoder=True)
        enc_h_cv = self._generate_conditions(batch, use_cvfeats=True, encoder=True)
        dec_h = self._generate_conditions(batch)
        dec_h_cv = self._generate_conditions(batch, use_cvfeats=True)
        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]

        # train discriminator
        outputs = self.model["G"].cycle_forward(
            feats,
            org_enc_h=enc_h,
            org_dec_h=dec_h,
            cv_enc_h=enc_h_cv,
            cv_dec_h=dec_h_cv,
        )
        loss = self.calculate_cyclediscriminator_loss(batch, outputs, loss)

        if phase == "train":
            self.optimizer["D"].zero_grad()
            loss["D"].backward()
            self.optimizer["D"].step()
        return loss

    def calculate_cycleadv_loss(self, batch, outputs, loss):
        mask = batch["mask"]
        for c in range(self.conf["n_cycles"]):
            for io in ["org", "cv"]:
                lbl = "{}cyc_{}".format(c, io)
                D_outputs = (
                    self.model["D"]
                    .forward(outputs[c][io]["decoded"].transpose(1, 2))
                    .transpose(1, 2)
                )
                if self.conf["acgan_flag"]:
                    D_outputs, spkr_cls = torch.split(D_outputs, [1, self.n_spkrs], dim=2)
                    D_outputs = D_outputs.masked_select(mask)
                    loss["ce_adv_{}".format(lbl)] = self.criterion["ce"](
                        spkr_cls.reshape(-1, spkr_cls.size(2)),
                        batch["{}_h_scalar".format(io)].reshape(-1),
                    )
                    loss["G"] += self.conf["alphas"]["ce"] * loss["ce_adv_{}".format(lbl)]
                loss["adv_{}".format(lbl)] = self.criterion["mse"](
                    D_outputs, torch.ones_like(D_outputs)
                )
                loss["G"] += self.conf["alphas"]["adv"] * loss["adv_{}".format(lbl)]
        return loss

    def calculate_cyclediscriminator_loss(self, batch, outputs, loss):
        mask = batch["mask"]
        for c in range(self.conf["n_cycles"]):

            def return_sample(x):
                return self.model["D"](x.transpose(1, 2)).transpose(1, 2)

            # get discriminator outputs
            lbl = "{}cyc".format(c)
            sample = {
                "real": return_sample(batch["feats"]),
                "org_fake": return_sample(outputs[c]["org"]["decoded"].detach()),
                "cv_fake": return_sample(outputs[c]["cv"]["decoded"].detach()),
            }

            if self.conf["acgan_flag"]:
                for k in sample.keys():
                    if k in ["real", "org_fake"]:
                        h_scalar = batch["org_h_scalar"]
                    else:
                        h_scalar = batch["cv_h_scalar"]
                    sample[k], spkr_cls = torch.split(
                        sample[k], [1, self.n_spkrs], dim=2
                    )
                    loss["ce_{}_{}".format(k, lbl)] = self.criterion["ce"](
                        spkr_cls.reshape(-1, spkr_cls.size(2)), h_scalar.reshape(-1),
                    )
                    loss["D"] += (
                        self.conf["alphas"]["ce"] * loss["ce_{}_{}".format(k, lbl)]
                    )

            real_sample = sample["real"].masked_select(mask)
            loss["real_{}".format(lbl)] = self.criterion["mse"](
                real_sample, torch.ones_like(real_sample)
            )
            fake_key = random.choice(["org_fake", "cv_fake"])
            fake_sample = sample[fake_key].masked_select(mask)
            loss["fake_{}".format(lbl)] = self.criterion["mse"](
                fake_sample, torch.zeros_like(fake_sample)
            )
            loss["D"] += (
                self.conf["alphas"]["fake"] * loss["fake_{}".format(lbl)]
                + self.conf["alphas"]["real"] * loss["real_{}".format(lbl)]
            )
        return loss
