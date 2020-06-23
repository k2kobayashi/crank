#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# griffin-lim.py
# Copyright (C) 2020 Wen-Chin HUANG
#
# Distributed under terms of the MIT license.
#

import json
import os
import sys
import argparse
import logging
import multiprocessing as mp
import numpy as np
import pysptk
import pyworld as pw
import scipy
from fastdtw import fastdtw

from pathlib import Path
import soundfile as sf
from parallel_wavegan.utils import find_files
from sprocket.speech import FeatureExtractor
from crank.net.trainer.dataset import read_feature
from crank.utils import load_yaml, open_featsscp
from crank.utils import low_cut_filter
from crank.feature import Feature

def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def calculate(file_list, gt_file_list, conf, spkr_conf, args, MCD):
    
    for i, cvt_path in enumerate(file_list):
        basename = get_basename(cvt_path)
        number, _, trgspk, _ = basename.split("_")
        trgspk = trgspk.split("-")[-1]
        
        # extract features from converted waveform
        x, fs = sf.read(str(cvt_path))
        x = np.array(x, dtype=np.float)
        x = low_cut_filter(x, fs, cutoff=70)
        fe = FeatureExtractor(
            analyzer="world",
            fs=conf["fs"],
            fftl=conf["fftl"],
            shiftms=conf["shiftms"],
            minf0=spkr_conf[trgspk]["minf0"],
            maxf0=spkr_conf[trgspk]["maxf0"],
        )
        cvt_f0, _, _ = fe.analyze(x)
        cvt_mcep = fe.mcep(dim=conf["mcep_dim"], alpha=conf["mcep_alpha"])

        # get ground truth features
        gt_mcep = read_feature(gt_file_list[f"{trgspk}_{number}"], "mcep")
        gt_f0 = read_feature(gt_file_list[f"{trgspk}_{number}"], "f0")

        # non-silence parts
        gt_idx = np.where(gt_f0>0)[0]
        gt_mcep = gt_mcep[gt_idx]
        cvt_idx = np.where(cvt_f0>0)[0]
        cvt_mcep = cvt_mcep[cvt_idx]

        # DTW
        _, path = fastdtw(cvt_mcep, gt_mcep, dist=scipy.spatial.distance.euclidean)
        twf = np.array(path).T
        cvt_mcep_dtw = cvt_mcep[twf[0]]
        gt_mcep_dtw = gt_mcep[twf[1]]

        # MCD 
        diff2sum = np.sum((cvt_mcep_dtw - gt_mcep_dtw)**2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        print('{} {}'.format(basename, mcd))
        MCD.append(mcd)

def main():
    
    parser = argparse.ArgumentParser(
        description="calculate MCD.")
    parser.add_argument('--conf', type=str, required=True,
                        help='Configuration file')
    parser.add_argument('--spkr_conf', type=str, required=True,
                        help='Speaker configuration file')
    parser.add_argument('--featdir', type=str, required=True,
                        help='Root directory of ground truth feature h5 files')
    parser.add_argument('--outwavdir', type=str, required=True,
                        help='Converted waveform directory')
    parser.add_argument("--n_jobs", default=40, type=int,
                        help="number of parallel jobs")
    args = parser.parse_args()
    
    # logging info
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) " "%(levelname)s: %(message)s",
    )
    
    # load configure files
    conf = load_yaml(args.conf)["feature"]
    spkr_conf = load_yaml(args.spkr_conf)
    
    # find files
    converted_files = sorted(find_files(args.outwavdir))

    # load ground truth scp
    scp = {}
    featdir = Path(args.featdir) / conf["label"]
    gt_feats = open_featsscp(featdir / "eval" / "feats.scp")
    
    # Get and divide list
    print("number of utterances = %d" % len(converted_files))
    file_lists = np.array_split(converted_files, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        MCD = manager.list()
        processes = []
        for f in file_lists:
            p = mp.Process(target=calculate, args=(f, gt_feats, conf, spkr_conf, args, MCD))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        mMCD = np.mean(np.array(MCD))
        print('Mean MCD: {:.2f}'.format(mMCD))

if __name__ == '__main__':
    main()
