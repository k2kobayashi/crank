#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# mosnet.py
# Copyright (C) 2020 Wen-Chin HUANG
#
# Distributed under terms of the MIT license.
#

import argparse
import logging
import os
import sys
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from crank.utils import load_yaml
from parallel_wavegan.utils import find_files
import speechmetrics

def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def main():
    parser = argparse.ArgumentParser(
        description='Use MOSnet to predict quality scores.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--outwavdir', type=str, required=True,
                        help='Converted waveform directory')
    parser.add_argument('--out', '-O', type=str,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) " "%(levelname)s: %(message)s",
    )

    # load converted files.
    converted_files = sorted(find_files(args.outwavdir))
    logging.info(f"number of utterances = {len(converted_files)}")

    # construct metric class
    metrics = speechmetrics.load('mosnet', None)
    
    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'w', encoding='utf-8')
    
    # actual calculation
    scores = {}
    for cvt_path in converted_files:
        basename = get_basename(cvt_path)
        number, srcspk, trgspk = basename.split("_")
        trgspk = trgspk.split("-")[-1]
        srcspk = srcspk.split("-")[-1]

        scores[f"{srcspk}-{trgspk}-{number}"] = metrics(cvt_path)["mosnet"][0][0]

    # summarize by pair
    pairwise_scores = {}
    for k, v in scores.items():
        srcspk, trgspk, _ = k.split("-")
        pair = srcspk + "-" + trgspk
        if not pair in pairwise_scores:
            pairwise_scores[pair] = []
        pairwise_scores[pair].append(v)

    for k in sorted(pairwise_scores.keys()):
        score_list = pairwise_scores[k]
        mean_score = float(sum(score_list)/len(score_list))
        out.write(f"{k} {mean_score:.3f}\n")

if __name__ == "__main__":
    main()
