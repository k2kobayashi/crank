#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Dataset

"""

import h5py
import numpy as np
import random
from pathlib import Path
from abc import abstractmethod
from multiprocessing import Manager
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self, conf, scp, phase="train", scaler=None, batch_len=5000,
    ):
        self.conf = conf
        self.h5list = list(scp[phase]["feats"].values())
        self.spkrlist = scp["train"]["spkrs"]
        self.scaler = scaler
        self.batch_len = batch_len

        if self.conf["feat_type"] == "mlfb":
            self.features = ["mlfb", "lcf0", "uv"]
        elif self.conf["feat_type"] == "mcep":
            self.features = ["mcep", "lcf0", "uv", "cap"]

        self.spkrdict = dict(zip(self.spkrlist, range(len(self.spkrlist))))
        self.n_spkrs = len(self.spkrdict)

        if self.conf["cache_dataset"]:
            self.manager = Manager()
            self.caches = self.manager.list()
            for _ in range(len(self.h5list)):
                self.caches += [None]

    @abstractmethod
    def mid_getitem(self, sample, idx, h5f):
        raise NotImplementedError("No mid_getitem function.")

    def __len__(self):
        return len(self.h5list)

    def __getitem__(self, idx):
        # use cache
        if self.conf["cache_dataset"] and self.caches[idx] is not None:
            return self.caches[idx]

        h5f = str(self.h5list[idx])

        # preprocess
        sample = self._pre_getitem(idx, h5f)

        # mid proecss defined in child
        try:
            sample = self.mid_getitem(sample, idx, h5f)
        except NotImplementedError:
            pass

        # postprocess
        sample = self._post_getitem(sample)

        # cache for next epoch
        if self.conf["cache_dataset"]:
            self.caches[idx] = sample

        return sample

    def _pre_getitem(self, idx, h5f):
        sample = {}
        sample = self._read_features(sample, h5f)
        sample["flbl"] = str(
            Path(Path(self.h5list[idx]).parent.stem) / Path(self.h5list[idx]).stem
        )
        sample["org_spkr_name"] = str(Path(h5f).parent.stem)
        sample["cv_spkr_name"] = random.choice(
            [s for s in list(self.spkrdict.keys()) if s != sample["org_spkr_name"]]
        )
        sample["flen"] = sample["feats"].shape[0]
        sample["mask"] = np.ones([sample["flen"]], dtype=bool)[:, np.newaxis]
        sample["cv_lcf0"] = convert_f0(
            self.scaler, sample["lcf0"], sample["org_spkr_name"], sample["cv_spkr_name"]
        )
        sample["org_spkr_num"] = int(self.spkrdict[sample["org_spkr_name"]])
        sample["org_h_onehot"], sample["org_h_scalar"] = self._get_spkrcode(
            sample["org_spkr_name"], sample["flen"]
        )
        sample["cv_h_onehot"], sample["cv_h_scalar"] = self._get_spkrcode(
            sample["cv_spkr_name"], sample["flen"]
        )
        return sample

    def _post_getitem(self, sample):
        if self.scaler is not None:
            sample = self._transform(sample)
        if self.conf["spec_augment"]:
            feats = sample["feats"]
            for i in range(self.conf["n_apply_spec_augment"]):
                feats = apply_tfmask(feats)
            sample["feats_sa"] = feats
        sample = self._zero_padding(sample)
        return sample

    def _read_features(self, sample, h5f):
        for k in self.features:
            if k == self.conf["feat_type"]:
                sample["feats"] = read_feature(h5f, ext=k)
            else:
                sample[k] = read_feature(h5f, ext=k)
        return sample

    def _transform(self, sample):
        for k in self.features:
            if k == self.conf["feat_type"]:
                sample["feats"] = self.scaler[k].transform(sample["feats"])
            elif k not in ["uv", "cap"]:
                sample[k] = self.scaler[k].transform(sample[k])
        return sample

    def _get_spkrcode(self, spkr_name, flen):
        spkr_num = int(self.spkrdict[spkr_name])
        h_scalar = (np.ones(flen) * spkr_num).astype(np.int32)
        h_onehot = create_one_hot(flen, self.n_spkrs, spkr_num)
        return h_onehot, h_scalar

    def _zero_padding(self, sample):
        dlen = self.batch_len - sample["flen"]
        p = random.choice(list(range(0, abs(dlen)))) if dlen < 0 else 0
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                if k in ["org_h_scalar", "cv_h_scalar"]:
                    sample[k] = padding(v, dlen, self.batch_len, value=-100, p=p)
                    sample[k] = sample[k].astype(np.long)
                elif k in ["mask"]:
                    sample[k] = padding(v, dlen, self.batch_len, value=False, p=p)
                    sample[k] = sample[k].astype(bool)
                else:
                    sample[k] = padding(v, dlen, self.batch_len, value=0.0, p=p)
                    sample[k] = sample[k].astype(np.float32)
                assert (
                    sample[k].shape[0] == self.batch_len
                ), "ERROR in padding: {}, dlen{}, p{}, v{}".format(
                    k, dlen, p, sample[k].shape[0]
                )
        return sample


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
