#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""Feature extractor.

Extract features such as mlfb and mcep.
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from sprocket.speech import FeatureExtractor, Synthesizer
from sprocket.util import HDF5
from parallel_wavegan.bin.preprocess import logmelfilterbank

from crank.utils import low_cut_filter, convert_continuos_f0, mlfb2wavf

EPS = 1e-10


class Feature(object):
    def __init__(self, h5_dir, conf, spkr_conf, gl_flag=False, synth_flag=False):
        self.h5_dir = Path(h5_dir)
        self.conf = conf
        self.sconf = spkr_conf
        self.feats = {}
        self.gl_flag = gl_flag
        self.synth_flag = synth_flag

    def analyze(self, wavf, gl_flag=False):
        self.fs, x, flbl = self._open_wavf(wavf)
        assert self.fs == self.conf["fs"]

        h5f = self.h5_dir / (flbl + ".h5")
        if not h5f.exists():
            logging.info("extract: {}".format(wavf))

            # analyze mlfb
            self._analyze_mlfb(wavf)
            if gl_flag:
                self._mlfb2wavf(flbl)

            # analyze world features, cf0, uv, then synthesize
            self._analyze_world_features(x)
            if self.synth_flag and self.conf["fftl"] != 256:
                self._synthesize_world_features(flbl)

            # save as hdf5
            self._save_hdf5(h5f)
        else:
            logging.info("h5 file already exists: {}".format(str(h5f)))

    def _save_hdf5(self, h5f):
        h5 = HDF5(h5f, "a")
        for k, v in self.feats.items():
            h5.save(v, ext=k)
        h5.close()

    def _open_wavf(self, wavf):
        # get file name (wav/spkr/file.wav -> spkr/file)
        flbl = Path(wavf).stem
        x, fs = sf.read(str(wavf))
        x = np.array(x, dtype=np.float)
        x = low_cut_filter(x, fs, cutoff=70)
        return fs, x, flbl

    def _analyze_world_features(self, x):
        feat = FeatureExtractor(
            analyzer="world",
            fs=self.conf["fs"],
            fftl=self.conf["fftl"],
            shiftms=self.conf["shiftms"],
            minf0=self.sconf["minf0"],
            maxf0=self.sconf["maxf0"],
        )
        # analyze world based features
        self.feats["f0"], self.feats["spc"], self.feats["ap"] = feat.analyze(x)
        self.feats["uv"], self.feats["cf0"] = convert_continuos_f0(self.feats["f0"])
        self.feats["lf0"] = np.log(self.feats["f0"] + EPS)
        self.feats["lcf0"] = np.log(self.feats["cf0"])

        if self.conf["fftl"] != 256:
            # NOTE: 256 fft_size sometimes causes errors
            self.feats["mcep"] = feat.mcep(
                dim=self.conf["mcep_dim"], alpha=self.conf["mcep_alpha"]
            )
            self.feats["npow"] = feat.npow()
            self.feats["cap"] = feat.codeap()
            cap = self.feats["cap"]
            self.feats["ccap"] = np.zeros(cap.shape)
            self.feats["cap_uv"] = np.zeros(cap.shape)
            for d in range(self.feats["ccap"].shape[-1]):
                cap[np.where(cap[:, d] == max(cap[:, d])), d] = 0.0
                (
                    self.feats["cap_uv"][:, d],
                    self.feats["ccap"][:, d],
                ) = convert_continuos_f0(cap[:, d])

    def _synthesize_world_features(self, flbl):
        # constract Synthesizer class
        synthesizer = Synthesizer(
            fs=self.conf["fs"], fftl=self.conf["fftl"], shiftms=self.conf["shiftms"]
        )

        # analysis/synthesis using F0, mcep, and ap
        anasyn = synthesizer.synthesis(
            self.feats["f0"],
            self.feats["mcep"],
            self.feats["ap"],
            alpha=self.conf["mcep_alpha"],
        )
        self.feats["x_anasyn"] = np.clip(anasyn, -1.0, 1.0)
        anasynf = self.h5_dir / (flbl + "_anasyn.wav")
        sf.write(str(anasynf), anasyn, self.conf["fs"])

    def _analyze_mlfb(self, wavf):
        # read wav file as float format
        x, fs = sf.read(str(wavf))
        self.feats["mlfb"] = logmelfilterbank(
            x,
            self.conf["fs"],
            hop_size=self.conf["hop_size"],
            fft_size=self.conf["fftl"],
            win_length=self.conf["fftl"],
            window="hann",
            num_mels=self.conf["mlfb_dim"],
            fmin=self.conf["fmin"],
            fmax=self.conf["fmax"],
            eps=EPS,
        )

    def _mlfb2wavf(self, flbl):
        glf = self.h5_dir / (flbl + "_gl.wav")
        mlfb2wavf(
            self.feats["mlfb"],
            glf,
            fs=self.conf["fs"],
            n_mels=self.conf["mlfb_dim"],
            fftl=self.conf["fftl"],
            hop_size=self.conf["hop_size"],
        )
