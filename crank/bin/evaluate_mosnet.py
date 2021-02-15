#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# evaluate_mosnet.py
# Copyright (C) 2020 Wen-Chin HUANG
#
# Distributed under terms of the MIT license.
#

import argparse
import logging
import sys

from pathlib import Path
import speechmetrics


def main():
    parser = argparse.ArgumentParser(
        description="Use MOSnet to predict quality scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outwavdir", type=str, help="Converted waveform directory")
    parser.add_argument(
        "--out",
        type=str,
        help="If omitted, then output to sys.stdout",
    )
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) " "%(levelname)s: %(message)s",
    )

    # load converted files.
    converted_files = sorted(list(Path(args.outwavdir).rglob("*.wav")))
    logging.info(f"number of utterances = {len(converted_files)}")

    # construct metric class
    metrics = speechmetrics.load("mosnet", None)

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, "w", encoding="utf-8")

    # actual calculation
    scores = {}
    for cv_path in converted_files:
        score = metrics(str(cv_path))["mosnet"][0][0]
        basename = cv_path.stem
        number, orgspk, tarspk = basename.split("_")
        tarspk = tarspk.split("-")[-1]
        orgspk = orgspk.split("-")[-1]

        scores[f"{orgspk}-{tarspk}-{number}"] = score

    # summarize by pair
    pairwise_scores = {}
    for k, v in scores.items():
        orgspk, tarspk, _ = k.split("-")
        pair = orgspk + " " + tarspk
        if pair not in pairwise_scores:
            pairwise_scores[pair] = []
        pairwise_scores[pair].append(v)

    for k in sorted(pairwise_scores.keys()):
        score_list = pairwise_scores[k]
        mean_score = float(sum(score_list) / len(score_list))
        out.write(f"{k} {mean_score:.3f}\n")


if __name__ == "__main__":
    main()
