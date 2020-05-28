#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Generate kaldi-like scp related files for crank

"""

import argparse
import random
import yaml
import logging
from pathlib import Path


def generate_scp(scpdir, spkr, wavfs, phase="train"):
    def write_lines(path, lines):
        with open(path, "a") as fp:
            for line in lines:
                fp.write("{}\n".format(line))

    wavscp, utt2spk, spk2utt = [], [], [spkr]
    for f in wavfs:
        # wavscp
        uid = spkr + "_" + f.stem
        path = str(f)
        wavscp.append("{} {}".format(uid, path))

        # utt2spk
        utt2spk.append("{} {}".format(uid, spkr))

        # spk2utt
        spk2utt.append(uid)
    spk2utt = [" ".join(spk2utt)]

    tdir = Path(scpdir) / phase
    tdir.mkdir(parents=True, exist_ok=True)

    write_lines(str(tdir / "wav.scp"), wavscp)
    write_lines(str(tdir / "utt2spk"), utt2spk)
    write_lines(str(tdir / "spk2utt"), spk2utt)


def create_spkr_yml(path, spkrs):
    spkr_yml = {}
    for spkr in spkrs:
        spkr_yml[spkr] = {"minf0": 40, "maxf0": 700, "npow": -20}
    with open(path, "w") as fp:
        yaml.dump(spkr_yml, fp)


def main():
    dcp = "generate scp and spkr files"
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument(
        "--shuffle", default=False, action="store_true", help="Randomize"
    )
    parser.add_argument("--wavdir", type=str, help="wav directory")
    parser.add_argument("--scpdir", type=str, help="scp directory")
    parser.add_argument("--spkr_yml", type=str, help="yml file for speaker params")
    parser.add_argument(
        "--dev_utterances", type=int, default=5, help="# of development utterances"
    )
    parser.add_argument(
        "--eval_utterances", type=int, default=0, help="# of development utterances"
    )
    parser.add_argument(
        "--eval_speakers", type=str, nargs="*", help="name of evaluation speakers"
    )
    args = parser.parse_args()

    spkrs = [s.name for s in sorted(list(Path(args.wavdir).iterdir()))]
    if not Path(args.spkr_yml).exists():
        create_spkr_yml(args.spkr_yml, spkrs)

    if not Path(args.scpdir).exists():
        for spkr in spkrs:
            spkrdir = Path(args.wavdir) / spkr
            wavfs = [f for f in sorted(spkrdir.glob("**/*.wav"))]

            if args.shuffle:
                wavfs = random.sample(wavfs, len(wavfs))

            if args.eval_speakers[0] == "":
                if args.eval_utterances == 0:
                    # overlap dev and eval
                    generate_scp(
                        args.scpdir, spkr, wavfs[: -args.dev_utterances], phase="train"
                    )
                    generate_scp(
                        args.scpdir, spkr, wavfs[-args.dev_utterances :], phase="dev"
                    )
                    generate_scp(
                        args.scpdir, spkr, wavfs[-args.dev_utterances :], phase="eval"
                    )
                else:
                    # not overlap dev and eval
                    de = args.dev_utterances + args.eval_utterances
                    generate_scp(args.scpdir, spkr, wavfs[:-de], phase="train")
                    generate_scp(
                        args.scpdir,
                        spkr,
                        wavfs[-de : -de + args.dev_utterances],
                        phase="dev",
                    )
                    generate_scp(
                        args.scpdir, spkr, wavfs[-args.eval_utterances :], phase="eval"
                    )
            else:
                # use eval spkr for eval
                if spkr not in args.eval_speakers:
                    generate_scp(
                        args.scpdir, spkr, wavfs[: -args.dev_utterances], phase="train"
                    )
                    generate_scp(
                        args.scpdir, spkr, wavfs[-args.dev_utterances :], phase="dev"
                    )
                else:
                    generate_scp(args.scpdir, spkr, wavfs, phase="eval")
    else:
        logging.info("scp directory is already exists")
        logging.info("If you want to generate new, please delete {}".format(args.scp))


if __name__ == "__main__":
    main()
