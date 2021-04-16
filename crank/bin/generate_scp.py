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
import logging
import random
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) " "%(levelname)s: %(message)s",
)


def generate_scp(tdir, spkr, wavfs):
    def write_lines(path, lines):
        with open(str(path), "a") as fp:
            for line in lines:
                fp.write("{}\n".format(line))
        logging.info("{} generated".format(str(path)))

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

    tdir.mkdir(parents=True, exist_ok=True)
    write_lines(tdir / "wav.scp", wavscp)
    write_lines(tdir / "utt2spk", utt2spk)
    write_lines(tdir / "spk2utt", spk2utt)


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
        "--eval_utterances", type=int, default=0, help="# of evaluate utterances"
    )
    parser.add_argument(
        "--eval_speakers", type=str, nargs="*", help="name of evaluation speakers"
    )
    args = parser.parse_args()

    spkrs = [s.name for s in sorted(list(Path(args.wavdir).iterdir()))]
    assert (
        len(spkrs) > 2
    ), f"No spkr directory found in {args.wavdir}. Please set wav files correctly."
    if not Path(args.spkr_yml).exists():
        create_spkr_yml(args.spkr_yml, spkrs)

    scpdir = Path(args.scpdir)
    n_dev = args.dev_utterances
    n_eval = args.eval_utterances

    if not scpdir.exists():
        for spkr in spkrs:
            spkrdir = Path(args.wavdir) / spkr
            wavfs = [f for f in sorted(spkrdir.glob("**/*.wav"))]

            if args.shuffle:
                wavfs = random.sample(wavfs, len(wavfs))

            if args.eval_speakers[0] == "":
                if args.eval_utterances == 0:
                    if args.dev_utterances != 0:
                        # overlap dev and eval
                        generate_scp(scpdir / "train", spkr, wavfs[:-n_dev])
                        generate_scp(scpdir / "dev", spkr, wavfs[-n_dev:])
                        generate_scp(scpdir / "eval", spkr, wavfs[-n_dev:])
                    else:
                        raise ValueError(
                            "You need to make non-zero either dev or eval."
                        )
                else:
                    if args.dev_utterances != 0:
                        # not overlap dev and eval
                        de = n_dev + n_eval
                        generate_scp(scpdir / "train", spkr, wavfs[:-de])
                        generate_scp(scpdir / "dev", spkr, wavfs[-de : -de + n_dev])
                        generate_scp(scpdir / "eval", spkr, wavfs[-n_eval:])
                    else:
                        # no dev
                        generate_scp(scpdir / "train", spkr, wavfs[:-n_eval])
                        generate_scp(scpdir / "dev", spkr, wavfs[:-n_eval])
                        generate_scp(scpdir / "eval", spkr, wavfs[-n_eval:])
            else:
                # use eval spkr for eval
                if spkr not in args.eval_speakers:
                    generate_scp(scpdir / "train", spkr, wavfs[:-n_dev])
                    generate_scp(scpdir / "dev", spkr, wavfs[-n_dev:])
                else:
                    generate_scp(scpdir / "eval", spkr, wavfs)
    else:
        logging.info("scp directory already exists: {}".format(args.scpdir))


if __name__ == "__main__":
    main()
