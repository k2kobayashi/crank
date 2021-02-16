#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
"""
Base Dataset

"""

import random
from multiprocessing import Manager
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        conf,
        scp,
        scaler,
        phase="train",
    ):
        self.conf = conf
        self.h5list = list(scp[phase]["feats"].values())
        self.spkrlist = scp["train"]["spkrs"]
        self.scaler = scaler
        self.batch_len = self.conf["batch_len"]

        self.features = [
            self.conf["input_feat_type"],
            self.conf["output_feat_type"],
        ]
        self.features += ["lcf0", "uv"]
        if "mcep" in self.features:
            self.features += ["cap"]
        if conf["use_raw"]:
            self.features += ["raw"]
        self.spkrdict = dict(zip(self.spkrlist, range(len(self.spkrlist))))
        self.n_spkrs = len(self.spkrdict)

        if self.conf["cache_dataset"]:
            self.manager = Manager()
            self.caches = self.manager.list()
            for _ in range(len(self.h5list)):
                self.caches += [None]

    def __len__(self):
        return len(self.h5list)

    def __getitem__(self, idx):
        if self.conf["cache_dataset"] and self.caches[idx] is not None:
            return self.caches[idx]

        # preprocess
        h5f = str(self.h5list[idx])
        sample = self._pre_getitem(idx, h5f)

        # middle process
        sample = self._middle_getitem(sample)

        # postprocess
        sample = self._post_getitem(sample)

        if self.conf["cache_dataset"]:
            self.caches[idx] = sample
        return sample

    def _pre_getitem(self, idx, h5f):
        """Read feature vectors"""
        sample = {}
        sample = self._read_features(sample, h5f)
        sample["flbl"] = str(
            Path(Path(self.h5list[idx]).parent.stem) / Path(self.h5list[idx]).stem
        )
        sample["org_spkr_name"] = str(Path(h5f).parent.stem)
        sample["cv_spkr_name"] = random.choice(
            [s for s in list(self.spkrdict.keys()) if s != sample["org_spkr_name"]]
        )
        sample["flen"] = sample[self.features[0]].shape[0]
        sample["mask"] = np.ones(sample["flen"], dtype=bool)[:, np.newaxis]
        sample["org_h_onehot"], sample["org_h"] = self._get_spkrcode(
            sample["org_spkr_name"], sample["flen"]
        )
        sample["cv_h_onehot"], sample["cv_h"] = self._get_spkrcode(
            sample["cv_spkr_name"], sample["flen"]
        )
        sample["cv_lcf0"] = convert_f0(
            self.scaler,
            sample["lcf0"],
            sample["org_spkr_name"],
            sample["cv_spkr_name"],
        )

        return sample

    def _middle_getitem(self, sample):
        """Apply normalization and some modification"""
        if self.scaler is not None:
            sample = self._transform(sample)
        if "mcep" in self.features and not self.conf["use_mcep_0th"]:
            sample["mcep_0th"] = sample["mcep"][..., :1]
            sample["mcep"] = sample["mcep"][..., 1:]
        if sample[self.conf["output_feat_type"]] == "excit":
            sample["excit"] = np.hstack(sample["lcf0"], sample["uv"], sample["cap"])
        if self.conf["spec_augment"]:
            raise NotImplementedError("SpecAugument currently disabled.")
            sample["spec_augment"] = self._spec_augment(
                sample[self.conf["input_feat_type"]]
            )
        sample = self._zero_padding(sample)
        for ed in [
            "encoder_mask",
            "decoder_mask",
            "cycle_encoder_mask",
            "cycle_decoder_mask",
        ]:
            sample[ed] = np.copy(sample["mask"])
        if self.conf["causal"]:
            er = self.conf["encoder_receptive_size"]
            dr = self.conf["decoder_receptive_size"]
            sample["encoder_mask"][:er] = False
            sample["decoder_mask"][: er + dr] = False
            sample["cycle_encoder_mask"][: er * 2 + dr] = False
            sample["cycle_decoder_mask"][: (er + dr) * 2] = False
        del sample["mask"]
        return sample

    def _post_getitem(self, sample):
        # TODO: input feature modification such as SpecAugument and noise augment
        sample["in_feats"] = sample[self.conf["input_feat_type"]]
        sample["out_feats"] = sample[self.conf["output_feat_type"]]
        # sample["in_mod"] = sample[self.conf["input_feat_type"]]
        del sample[self.conf["input_feat_type"]]
        if self.conf["output_feat_type"] in sample.keys():
            del sample[self.conf["output_feat_type"]]
        return sample

    def _read_features(self, sample, h5f):
        for k in self.features:
            sample[k] = read_feature(h5f, ext=k)
        return sample

    def _transform(self, sample):
        for k in self.features:
            if k not in ["uv", "cap"] and k not in self.conf["ignore_scaler"]:
                sample[k] = self.scaler[k].transform(sample[k])
        return sample

    def _get_spkrcode(self, spkr_name, flen):
        spkr_num = int(self.spkrdict[spkr_name])
        h = (np.ones(flen) * spkr_num).astype(np.long)
        h_onehot = create_one_hot(flen, self.n_spkrs, spkr_num)
        return h_onehot, h

    def _zero_padding(self, sample):
        blen = self.batch_len
        diff_frames = blen - sample["flen"]
        offset = self.conf["feature"]["fftl"] // self.conf["feature"]["hop_size"]
        if diff_frames < 0 and abs(diff_frames) > offset * 2:
            # length of sample is larger than batch_len
            p = random.choice(range(offset, abs(diff_frames) - offset))
        else:
            # use from its begining
            p = 0
        for k, v in sample.items():
            if not isinstance(v, np.ndarray):
                continue

            if k in ["org_h", "cv_h"]:
                # padding -100 for ignore_index
                sample[k] = padding(v, diff_frames, blen, value=-100, p=p).astype(
                    np.long
                )
            elif k in ["mask"]:
                sample[k] = padding(v, diff_frames, blen, value=False, p=p).astype(bool)
            elif k in ["raw"]:
                # padding 0 for raw samples
                sample[k] = padding_raw(
                    v.squeeze(),
                    diff_frames,
                    blen,
                    self.conf["feature"]["fftl"],
                    self.conf["feature"]["hop_size"],
                    value=0.0,
                    p=p,
                ).astype(np.float)
            else:
                # padding 0 for continuous values
                sample[k] = padding(v, diff_frames, blen, value=0.0, p=p).astype(
                    np.float32
                )
            if k not in ["raw"]:
                assert (
                    sample[k].shape[0] == blen
                ), "ERROR in padding: {}, diff_frames{}, p{}, v{}".format(
                    k, diff_frames, p, sample[k].shape[0]
                )
        return sample

    def _spec_augment(self, feats):
        for i in range(self.conf["n_spec_augment"]):
            feats = apply_tfmask(feats)
        return feats


def apply_tfmask(feats, max_bin=27, max_time=100):
    """Apply time-frequency mask SpecAugument"""
    flen, dim = feats.shape
    d_mask = random.randint(1, max_bin)
    d_point = random.randint(0, dim - d_mask)
    t_mask = random.randint(1, max_time)
    t_point = random.randint(0, flen - t_mask)

    mask = np.ones((flen, dim), dtype="d")
    mask[:, d_point : d_point + d_mask] = 0
    mask[t_point : t_point + t_mask] = 0
    return feats * mask


def create_one_hot(T, N, c, B=-1):
    if B == -1:
        y = np.zeros((T, N), dtype=np.float32)
        y[:, c] = 1
    else:
        y = np.zeros((B, T, N), dtype=np.float32)
        y[:, :, c] = 1
    return y


def read_feature(h5f, ext="mlfb"):
    with h5py.File(h5f, "r") as fp:
        data = fp[ext][:]
    if len(data.shape) == 1:
        return data[:, np.newaxis]
    else:
        return data


def padding(x, dlen, batch_len, value=0.0, p=0):
    if dlen >= 0:
        # padding
        actual_dlen = batch_len - x.shape[0]
        if actual_dlen != 0:
            if len(x.shape) == 2:
                x = np.concatenate([x, np.ones((actual_dlen, x.shape[1])) * value])
            elif len(x.shape) == 1:
                x = np.concatenate([x, np.ones((actual_dlen)) * value])
        else:
            return x
    elif dlen < 0:
        # discard
        x = x[p : p + batch_len]
    return x


def padding_raw(x, dlen, batch_len, fftl, hop_size, value=0.0, p=0):
    target_length = fftl + hop_size * batch_len - 1

    if dlen > 0:
        # padding
        if len(x) < target_length - fftl:
            x = np.pad(x, int(fftl // 2), mode="reflect")
        if len(x) < target_length:
            x = np.concatenate([x, np.zeros(target_length - len(x))])
    else:
        ph = p * hop_size
        hfftl = fftl // 2
        if p == 0:
            x = np.concatenate([np.zeros(hfftl), x])
            ph = hfftl + 1
        # discard
        assert ph > hfftl, "{}, {}, {}".format(ph, hfftl, p)

        require_length = ph + target_length
        if len(x) < require_length:
            x = np.concatenate([x, np.zeros(require_length - len(x))])
        x = x[ph - hfftl : ph + hfftl + hop_size * batch_len - 1]
        # print(len(x), target_length, require_length, dlen, p)
    assert len(x) == target_length
    return x


def calculate_maxflen(flist):
    max_flen = 0
    for h5f in flist:
        with h5py.File(h5f, "r") as fp:
            if max_flen < fp["mlfb"].shape[0]:
                max_flen = fp["mlfb"].shape[0]
    return max_flen


def convert_f0(scaler, lcf0, org_spkr_name, cv_spkr_name):
    return (lcf0 - scaler[org_spkr_name]["lcf0"].mean_) / np.sqrt(
        scaler[org_spkr_name]["lcf0"].var_
    ) * np.sqrt(scaler[cv_spkr_name]["lcf0"].var_) + scaler[cv_spkr_name]["lcf0"].mean_
