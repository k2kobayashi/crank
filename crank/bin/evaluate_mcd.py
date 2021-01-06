#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# evaluate_mcd.py
# Copyright (C) 2020 Wen-Chin HUANG
#
# Distributed under terms of the MIT license.
#

import sys
import argparse
import logging
import numpy as np
import scipy
from fastdtw import fastdtw
from joblib import Parallel, delayed

from pathlib import Path
import soundfile as sf
from sprocket.speech import FeatureExtractor
from crank.net.trainer.dataset import read_feature
from crank.utils import load_yaml, open_featsscp
from crank.utils import low_cut_filter


def get_world_features(wavpath, spk, conf, spkr_conf):
    x, fs = sf.read(str(wavpath))
    x = np.array(x, dtype=np.float)
    x = low_cut_filter(x, fs, cutoff=70)
    fe = FeatureExtractor(
        analyzer="world",
        fs=conf["feature"]["fs"],
        fftl=conf["feature"]["fftl"],
        shiftms=conf["feature"]["shiftms"],
        minf0=spkr_conf[spk]["minf0"],
        maxf0=spkr_conf[spk]["maxf0"],
    )
    cv_f0, _, _ = fe.analyze(x)
    cv_mcep = fe.mcep(
        dim=conf["feature"]["mcep_dim"], alpha=conf["feature"]["mcep_alpha"]
    )
    return cv_mcep, cv_f0


def calculate(cv_path, gt_file_list, conf, spkr_conf):

    basename = cv_path.stem
    number, orgspk, tarspk = basename.split("_")
    tarspk = tarspk.split("-")[-1]
    orgspk = orgspk.split("-")[-1]

    # get converted features. If mcep, from h5; else waveform
    if conf["output_feat_type"] == "mcep":
        cv_mcep = read_feature(cv_path, "feat")
        cv_f0 = read_feature(cv_path, "f0")
    else:
        cv_mcep, cv_f0 = get_world_features(cv_path, tarspk, conf, spkr_conf)

    # get ground truth features
    gt_mcep = read_feature(gt_file_list[f"{tarspk}_{number}"], "mcep")
    gt_f0 = read_feature(gt_file_list[f"{tarspk}_{number}"], "f0")

    # non-silence parts
    gt_idx = np.where(gt_f0 > 0)[0]
    gt_mcep = gt_mcep[gt_idx]
    cv_idx = np.where(cv_f0 > 0)[0]
    cv_mcep = cv_mcep[cv_idx]

    # DTW
    _, path = fastdtw(cv_mcep, gt_mcep, dist=scipy.spatial.distance.euclidean)
    twf = np.array(path).T
    cv_mcep_dtw = cv_mcep[twf[0]]
    gt_mcep_dtw = gt_mcep[twf[1]]

    # MCD
    diff2sum = np.sum((cv_mcep_dtw - gt_mcep_dtw) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    return f"{orgspk}-{tarspk}-{number}", mcd


def main():

    parser = argparse.ArgumentParser(description="calculate MCD.")
    parser.add_argument("--conf", type=str, help="configuration file")
    parser.add_argument("--spkr_conf", type=str, help="speaker configuration file")
    parser.add_argument(
        "--featdir", type=str, help="root directory of ground truth h5",
    )
    parser.add_argument("--outwavdir", type=str, help="converted waveform directory")
    parser.add_argument(
        "--out", type=str, help="if omitted, then output to sys.stdout",
    )
    parser.add_argument("--n_jobs", default=1, type=int, help="number of parallel jobs")
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
    if conf["output_feat_type"] == "mcep":
        converted_files = sorted(list(Path(args.outwavdir).glob("*.h5")))
    else:
        converted_files = sorted(list(Path(args.outwavdir).rglob("*.wav")))
    logging.info(f"number of utterances = {len(converted_files)}")

    # load ground truth scp
    featdir = Path(args.featdir) / conf["feature"]["label"]
    gt_feats = open_featsscp(featdir / "eval" / "feats.scp")

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, "w", encoding="utf-8")

    MCD_list = Parallel(args.n_jobs)(
        [
            delayed(calculate)(cv_path, gt_feats, conf, spkr_conf)
            for cv_path in converted_files
        ]
    )

    # summarize by pair
    pairwise_MCD = {}
    for k, v in MCD_list:
        orgspk, tarspk, _ = k.split("-")
        pair = orgspk + " " + tarspk
        if pair not in pairwise_MCD:
            pairwise_MCD[pair] = []
        pairwise_MCD[pair].append(v)

    for k in sorted(pairwise_MCD.keys()):
        mcd_list = pairwise_MCD[k]
        mean_mcd = float(sum(mcd_list) / len(mcd_list))
        out.write(f"{k} {mean_mcd:.3f}\n")


if __name__ == "__main__":
    main()
