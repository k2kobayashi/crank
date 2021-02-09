#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. Kobayashi <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
"""
StarGAN trainer

"""

import random

from crank.net.trainer import LSGANTrainer


class StarGANTrainer(LSGANTrainer):
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

    def update_G(self, batch, loss, phase="train"):
        enc_h = self._get_enc_h(batch)
        enc_h_cv = self._get_enc_h(batch, use_cvfeats=True)
        dec_h, spkrvec = self._get_dec_h(batch)
        dec_h_cv, spkrvec_cv = self._get_dec_h(batch, use_cvfeats=True)
        feats = batch["in_feats"]
        cycle_outputs = self.model["G"].cycle_forward(
            feats, enc_h, dec_h, enc_h_cv, dec_h_cv, spkrvec, spkrvec_cv
        )
        if self.conf["use_vqvae_loss"]:
            loss = self.calculate_vqvae_loss(batch, cycle_outputs[0]["org"], loss)
        loss = self.calculate_cyclevqvae_loss(batch, cycle_outputs, loss)
        if self.conf["use_spkradv_training"]:
            for label in ["cv", "recon"]:
                loss = self.calculate_spkradv_loss(
                    batch, cycle_outputs[0][label], loss, label=label, phase=phase
                )

        # adv loss
        loss = self.calculate_adv_loss(
            batch,
            cycle_outputs[0]["cv"]["decoded"],
            batch["cv_h"],
            batch["decoder_mask"],
            loss,
        )

        if phase == "train" and not self.stop_generator:
            self.step_model(loss, model="G")
        return loss

    def update_D(self, batch, loss, phase="train"):
        def return_sample(x):
            return self.model["D"](x.transpose(1, 2)).transpose(1, 2)

        enc_h_cv = self._get_enc_h(batch, use_cvfeats=True)
        dec_h_cv, spkrvec_cv = self._get_dec_h(batch, use_cvfeats=True)
        feats = batch["in_feats"]

        if self.conf["switch_update"]:
            updates = random.choice(["real", "fake"])
        else:
            updates = ["real", "fake"]

        real_inputs = self.get_D_inputs(batch, batch["in_feats"], label="org")
        loss = self.calculate_discriminator_loss(
            return_sample(real_inputs),
            batch["org_h"],
            batch["decoder_mask"],
            loss,
            label="real",
            updates=updates,
        )

        outputs = self.model["G"].forward(feats, enc_h_cv, dec_h_cv, spkrvec_cv)
        fake_inputs = self.get_D_inputs(batch, outputs["decoded"].detach(), label="cv")
        loss = self.calculate_discriminator_loss(
            return_sample(fake_inputs),
            batch["cv_h"],
            batch["decoder_mask"],
            loss,
            label="fake",
            updates=updates,
        )

        if phase == "train":
            self.step_model(loss, model="D")
        return loss
