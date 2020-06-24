#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# mcd_calculate.py
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

def get_world_features(cvt_path, trgspk, conf, spkr_conf):
    x, fs = sf.read(str(cvt_path))
    x = np.array(x, dtype=np.float)
    x = low_cut_filter(x, fs, cutoff=70)
    fe = FeatureExtractor(
        analyzer="world",
        fs=conf["feature"]["fs"],
        fftl=conf["feature"]["fftl"],
        shiftms=conf["feature"]["shiftms"],
        minf0=spkr_conf[trgspk]["minf0"],
        maxf0=spkr_conf[trgspk]["maxf0"],
    )
    cvt_f0, _, _ = fe.analyze(x)
    cvt_mcep = fe.mcep(dim=conf["feature"]["mcep_dim"],
                       alpha=conf["feature"]["mcep_alpha"])
    return cvt_mcep, cvt_f0

def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def calculate(file_list, gt_file_list, conf, spkr_conf, MCD):
    
    for i, cvt_path in enumerate(file_list):
        basename = get_basename(cvt_path)
        number, srcspk, trgspk = basename.split("_")
        trgspk = trgspk.split("-")[-1]
        srcspk = srcspk.split("-")[-1]
        
        # get converted features. If mcep, from h5; else waveform
        if conf["feat_type"] == "mcep":
            cvt_mcep = read_feature(cvt_path, "feat")
            cvt_f0 = read_feature(cvt_path, "f0")
        else:
            cvt_mcep, cvt_f0 = get_world_features(cvt_path, trgspk, conf, spkr_conf)

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
        MCD[f"{srcspk}-{trgspk}-{number}"] = mcd

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
    parser.add_argument('--out', '-O', type=str,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')
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
    conf = load_yaml(args.conf)
    spkr_conf = load_yaml(args.spkr_conf)
    
    # load converted files. If mcep, use h5; else, waveform
    if conf["feat_type"] == "mcep":
        converted_files = sorted(find_files(args.outwavdir, query="*.h5"))
    else:
        converted_files = sorted(find_files(args.outwavdir))

    # load ground truth scp
    scp = {}
    featdir = Path(args.featdir) / conf["feature"]["label"]
    gt_feats = open_featsscp(featdir / "eval" / "feats.scp")
    
    # Get and divide list
    logging.info(f"number of utterances = {len(converted_files)}")
    file_lists = np.array_split(converted_files[:100], args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]
    
    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'w', encoding='utf-8')

    # multi processing
    with mp.Manager() as manager:
        MCD = manager.dict()
        processes = []
        for f in file_lists:
            p = mp.Process(target=calculate, args=(f, gt_feats, conf, spkr_conf, MCD))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        # summarize by pair
        pairwise_MCD = {}
        for k, v in MCD.items():
            srcspk, trgspk, _ = k.split("-")
            pair = srcspk + "-" + trgspk
            if not pair in pairwise_MCD:
                pairwise_MCD[pair] = []
            pairwise_MCD[pair].append(v)

    for k in sorted(pairwise_MCD.keys()):
        mcd_list = pairwise_MCD[k]
        mean_mcd = float(sum(mcd_list)/len(mcd_list))
        out.write(f"{k} {mean_mcd:.3f}\n")

if __name__ == '__main__':
    main()
