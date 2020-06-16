#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# griffin-lim.py
# Copyright (C) 2020 Wen-Chin HUANG
#
# Distributed under terms of the MIT license.
#

import argparse
import logging
import os
import sys

import numpy as np


from crank.utils import mlfb2wavf
from crank.utils import load_yaml, open_scpdir, open_featsscp
from crank.net.trainer.dataset import read_feature

from parallel_wavegan.utils import find_files

from joblib import Parallel, delayed


def main():
    parser = argparse.ArgumentParser(
        description='Convert filter banks to waveform using Griffin-Lim algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--conf', type=str, required=True,
                        help='Cofiguration file')
    parser.add_argument('--rootdir', type=str, required=True,
                        help='Root directory of filter bank h5 files')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) " "%(levelname)s: %(message)s",
    )
    
    # load configure files
    conf = load_yaml(args.conf)
    for k, v in conf.items():
        logging.info("{}: {}".format(k, v))

    # find h5 files
    feats_files = find_files(args.rootdir, "*.h5")
    feats = {
        os.path.join(args.outdir, os.path.basename(filename).replace(".h5", ".wav")):read_feature(filename, "feats")
        for filename in feats_files
    }

    # Main Griffin-Lim algorithm
    Parallel(n_jobs=30)(
        [
            delayed(mlfb2wavf)(
                feats[wavf],
                wavf,
                fs=conf["feature"]["fs"],
                n_mels=conf["feature"]["mlfb_dim"],
                fftl=conf["feature"]["fftl"],
                hop_size=conf["feature"]["hop_size"],
                plot=False,
            )
            for wavf in list(feats.keys())
        ]
    )

if __name__ == "__main__":
    main()
