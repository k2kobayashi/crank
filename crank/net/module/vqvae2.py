#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
VQVAE class

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from parallel_wavegan.models import ParallelWaveGANGenerator


class VQVAE2(nn.Module):
    def __init__(self, conf, spkr_size=0):
        super(VQVAE2, self).__init__()
        self.conf = conf
        self.spkr_size = spkr_size
        self._construct_net()

        if self.conf["use_spkr_embedding"]:
            self.spkr_embedding = nn.Embedding(
                self.spkr_size, self.conf["spkr_embedding_size"]
            )
            if self.conf["use_embedding_transform"]:
                self.embedding_transform = nn.Linear(
                    self.conf["spkr_embedding_size"],
                    self.conf["embedding_transform_size"],
                )

    def forward(
        self, x, enc_h, dec_h, spkrvec=None, use_ema=True, encoder_detach=False
    ):
        x = x.transpose(1, 2)
        if spkrvec is not None:
            spkremb = self.spkr_embedding(spkrvec)
            if dec_h is not None:
                dec_h = torch.cat([dec_h, spkremb], axis=-1)
            else:
                dec_h = spkremb
        enc_h = enc_h.transpose(1, 2) if enc_h is not None else None
        dec_h = dec_h.transpose(1, 2) if dec_h is not None else None

        enc, spkr_cls = self.encode(x, enc_h=enc_h)
        enc, dec, emb_idxs, _, qidxs = self.decode(
            enc, dec_h, use_ema=use_ema, detach=encoder_detach
        )
        outputs = self.make_dict(enc, dec, emb_idxs, qidxs, spkr_cls)
        return outputs

    def cycle_forward(
        self, x, org_enc_h, org_dec_h, cv_enc_h, cv_dec_h, org_spkrvec, cv_spkrvec
    ):
        x = x.transpose(1, 2)
        if org_spkrvec is not None:
            org_spkremb = self.spkr_embedding(org_spkrvec)
            if org_dec_h is not None:
                org_dec_h = torch.cat([org_dec_h, org_spkremb], axis=-1)
            else:
                org_dec_h = org_spkremb
        if cv_spkrvec is not None:
            cv_spkremb = self.spkr_embedding(cv_spkrvec)
            if cv_dec_h is not None:
                cv_dec_h = torch.cat([cv_dec_h, cv_spkremb], axis=-1)
            else:
                cv_dec_h = cv_spkremb
        org_enc_h = org_enc_h.transpose(1, 2) if org_enc_h is not None else None
        org_dec_h = org_dec_h.transpose(1, 2) if org_dec_h is not None else None
        cv_enc_h = cv_enc_h.transpose(1, 2) if cv_enc_h is not None else None
        cv_dec_h = cv_dec_h.transpose(1, 2) if cv_dec_h is not None else None

        outputs = []
        for n in range(self.conf["n_cycles"]):
            enc, org_spkr_cls = self.encode(x, enc_h=org_enc_h)
            org_enc, org_dec, org_emb_idxs, _, org_qidxs = self.decode(
                enc, org_dec_h, use_ema=True
            )
            cv_enc, cv_dec, cv_emb_idxs, _, cv_qidxs = self.decode(
                enc, cv_dec_h, use_ema=False
            )

            enc, cv_spkr_cls = self.encode(cv_dec, enc_h=cv_enc_h)
            recon_enc, recon_dec, recon_emb_idxs, _, recon_qidxs = self.decode(
                enc, org_dec_h, use_ema=True
            )
            outputs.append(
                {
                    "org": self.make_dict(
                        org_enc, org_dec, org_emb_idxs, org_qidxs, org_spkr_cls
                    ),
                    "cv": self.make_dict(
                        cv_enc, cv_dec, cv_emb_idxs, cv_qidxs, cv_spkr_cls,
                    ),
                    "recon": self.make_dict(
                        recon_enc, recon_dec, recon_emb_idxs, recon_qidxs, None,
                    ),
                }
            )
            x = recon_dec.detach()
        return outputs

    def encode(self, x, enc_h=None):
        # encode
        encoded = []
        for n in range(self.conf["n_vq_stacks"]):
            if n == 0:
                enc = self.encoders[n](x, c=enc_h)
                spkr_cls = None
                if self.conf["encoder_spkr_classifier"]:
                    enc, spkr_cls = torch.split(
                        enc, [self.conf["emb_dim"][n], self.spkr_size], dim=1
                    )
            else:
                enc = self.encoders[n](enc, c=None)
            encoded.append(enc)
        return encoded, spkr_cls

    def decode(self, enc, dec_h, use_ema=True, detach=False):
        # decode
        dec = 0
        emb_idxs, emb_idx_qxs, qidxs = [], [], []
        for n in reversed(range(self.conf["n_vq_stacks"])):
            # vq
            enc[n] = enc[n] + dec
            emb_idx, emb_idx_qx, qidx = self.quantizers[n](enc[n], use_ema=use_ema)
            if detach:
                emb_idx_qx = emb_idx_qx.detach()
            emb_idxs.append(emb_idx)
            emb_idx_qxs.append(emb_idx_qx)
            qidxs.append(qidx)

            # decode
            if n != 0:
                dec = self.decoders[n](emb_idx_qx, c=None)
            else:
                dec = self.decoders[n](torch.cat(emb_idx_qxs, dim=1), c=dec_h)
        return enc, dec, emb_idxs, emb_idx_qxs, qidxs

    def remove_weight_norm(self):
        for n in range(self.conf["n_vq_stacks"]):
            self.encoders[n].remove_weight_norm()
            self.decoders[n].remove_weight_norm()

    def make_dict(self, enc, dec, emb_idxs, qidxs, spkr_cls=None):
        # NOTE: transpose from [B, D, T] to be [B, T, D]
        # NOTE: index of bottom outputs to be 0
        return {
            "encoded": [e.transpose(1, 2) for e in enc],
            "spkr_cls": spkr_cls.transpose(1, 2) if spkr_cls is not None else None,
            "decoded": dec.transpose(1, 2),
            "emb_idx": emb_idxs[::-1],
            "qidx": qidxs[::-1],
        }

    def _construct_net(self):
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.quantizers = nn.ModuleList()
        for n in range(self.conf["n_vq_stacks"]):
            if n == 0:
                enc_in_channels = self.conf["input_size"]
                enc_out_channels = self.conf["emb_dim"][n]
                if self.conf["encoder_spkr_classifier"]:
                    enc_out_channels += self.spkr_size
                enc_aux_channels = self.conf["enc_aux_size"]
                dec_in_channels = sum(
                    [self.conf["emb_dim"][i] for i in range(self.conf["n_vq_stacks"])]
                )
                dec_out_channels = self.conf["output_size"]
                if self.conf["use_spkr_embedding"]:
                    if not self.conf["use_embedding_transform"]:
                        dec_aux_channels = (
                            self.conf["dec_aux_size"] + self.conf["spkr_embedding_size"]
                        )
                    else:
                        dec_aux_channels = (
                            self.conf["dec_aux_size"]
                            + self.conf["embedding_transform_size"]
                        )
                else:
                    dec_aux_channels = self.conf["dec_aux_size"] + self.spkr_size
            elif n >= 1:
                enc_in_channels = self.conf["emb_dim"][n - 1]
                enc_out_channels = self.conf["emb_dim"][n]
                enc_aux_channels = 0
                dec_in_channels = self.conf["emb_dim"][n]
                dec_out_channels = self.conf["emb_dim"][n - 1]
                dec_aux_channels = 0
            self.encoders.append(
                ParallelWaveGANGenerator(
                    in_channels=enc_in_channels,
                    out_channels=enc_out_channels,
                    kernel_size=self.conf["kernel_size"][n],
                    layers=self.conf["n_layers"][n] * self.conf["n_layers_stacks"][n],
                    stacks=self.conf["n_layers_stacks"][n],
                    residual_channels=self.conf["residual_channels"],
                    gate_channels=128,
                    skip_channels=64,
                    aux_channels=enc_aux_channels,
                    aux_context_window=0,
                    dropout=0.0,
                    bias=True,
                    use_weight_norm=True,
                    use_causal_conv=self.conf["causal"],
                    upsample_conditional_features=False,
                )
            )
            self.decoders.append(
                ParallelWaveGANGenerator(
                    in_channels=dec_in_channels,
                    out_channels=dec_out_channels,
                    kernel_size=self.conf["kernel_size"][n],
                    layers=self.conf["n_layers"][n] * self.conf["n_layers_stacks"][n],
                    stacks=self.conf["n_layers_stacks"][n],
                    residual_channels=self.conf["residual_channels"],
                    gate_channels=128,
                    skip_channels=64,
                    aux_channels=dec_aux_channels,
                    aux_context_window=0,
                    dropout=0.0,
                    bias=True,
                    use_weight_norm=True,
                    use_causal_conv=self.conf["causal"],
                    upsample_conditional_features=False,
                )
            )
            self.quantizers.append(
                Quantizer(
                    self.conf["emb_dim"][n],
                    self.conf["emb_size"][n],
                    ema_flag=self.conf["ema_flag"],
                    bdt_flag=True,
                )
            )


class Quantizer(nn.Module):
    def __init__(
        self, emb_dim, emb_size, decay=0.99, eps=1e-5, ema_flag=False, bdt_flag=False
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_size = emb_size
        self.ema_flag = ema_flag
        self.bdt_flag = bdt_flag  # use (B, D, T) instead of (B, T, D)

        self.embedding = nn.Embedding(emb_size, emb_dim)  # (K, D)
        self.embedding.weight.data.uniform_(-1.0 / self.emb_size, 1.0 / self.emb_size)

        if self.ema_flag:
            self.decay = decay
            self.eps = eps
            embed = torch.randn(emb_dim, emb_size)
            self.register_buffer("ema_size", torch.zeros(emb_size))
            self.register_buffer("ema_w", embed.clone())

    def forward(self, x, use_ema=True):
        if self.bdt_flag:
            x = x.transpose(1, 2)

        quantized_idx, quantized_onehot = self.vq(x)
        embed_idx = torch.matmul(
            quantized_onehot.float(), self.embedding.weight
        )  # (B, T)
        # exponential moving average
        if self.training and self.ema_flag and use_ema:
            self.ema_size = self.decay * self.ema_size + (1 - self.decay) * torch.sum(
                quantized_onehot.view(-1, self.emb_size), 0
            )
            embed_sum = torch.sum(
                torch.matmul(x.transpose(1, 2), quantized_onehot.float()), dim=0
            )
            self.ema_w.data = (
                self.decay * self.ema_w.data + (1 - self.decay) * embed_sum
            )
            n = torch.sum(self.ema_size)
            self.ema_size = (
                (self.ema_size + self.eps) / (n + self.emb_size * self.eps) * n
            )
            embed_normalized = self.ema_w / self.ema_size.unsqueeze(0)
            self.embedding.weight.data.copy_(embed_normalized.transpose(0, 1))

        # connect graph
        embed_idx_qx = x + (embed_idx - x).detach()
        if self.bdt_flag:
            embed_idx_qx = embed_idx_qx.transpose(1, 2)
        return embed_idx, embed_idx_qx, quantized_idx

    def vq(self, x):
        flatten_x = x.reshape(-1, self.emb_dim)
        dist = (
            torch.sum(torch.pow(self.embedding.weight, 2), dim=1)
            - 2 * torch.matmul(flatten_x, self.embedding.weight.T)
            + torch.sum(torch.pow(flatten_x, 2), dim=1, keepdim=True)
        )
        quantized_idx = torch.argmin(dist, dim=1).view(x.size(0), x.size(1))
        quantized_onehot = F.one_hot(quantized_idx, self.emb_size)
        return quantized_idx, quantized_onehot
