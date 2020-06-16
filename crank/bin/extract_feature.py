#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Extract various features

"""

import argparse
import logging
from pathlib import Path
from joblib import Parallel, delayed

from crank.feature import Feature
from crank.utils import load_yaml, open_scpdir

logging.basicConfig(level=logging.INFO)


def main():
    dcp = "Extract aoucstic features of the speaker"
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument("--n_jobs", type=int, default=-1, help="# of CPUs")
    parser.add_argument("--phase", type=str, default=None, help="phase")
    parser.add_argument("--conf", type=str, help="ymal file for network parameters")
    parser.add_argument("--spkr_yml", type=str, help="yml file for speaker params")
    parser.add_argument("--scpdir", type=str, help="scp directory")
    parser.add_argument("--featdir", type=str, help="output feature directory")
    args = parser.parse_args()

    conf = load_yaml(args.conf)
    spkr_conf = load_yaml(args.spkr_yml)
    scp = open_scpdir(Path(args.scpdir) / args.phase)

    featdir = Path(args.featdir) / conf["feature"]["label"] / args.phase
    featsscp = featdir / "feats.scp"
    (featsscp).unlink(missing_ok=True)

    for spkr in scp["spkrs"]:
        logging.info("extract feature for {}".format(spkr))
        wavs = [scp["wav"][uid] for uid in scp["spk2utt"][spkr]]
        (featdir / spkr).mkdir(parents=True, exist_ok=True)
        feat = Feature(featdir / spkr, conf["feature"], spkr_conf[spkr])

        # create feats.scp
        with open(featsscp, "a") as fp:
            for uid in scp["spk2utt"][spkr]:
                wavf = scp["wav"][uid]
                h5f = str(featdir / spkr / (Path(wavf).stem + ".h5"))
                fp.write("{} {}\n".format(uid, h5f))

        # feature extraction with GliffinLim
        Parallel(n_jobs=args.n_jobs)(
            [
                delayed(feat.analyze)(wavf, gl_flag=True)
                for wavf in wavs[: conf["n_gl_samples"]]
            ]
        )

        # feature extraction without GliffinLim
        Parallel(n_jobs=args.n_jobs)(
            [
                delayed(feat.analyze)(wavf, gl_flag=False)
                for wavf in wavs[conf["n_gl_samples"] :]
            ]
        )


if __name__ == "__main__":
    main()
