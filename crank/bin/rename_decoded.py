#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2021 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import argparse
from pathlib import Path


def main():
    # options for python
    description = "Rename decoded waveforms"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--outwavdir", type=str, help="decoded waveform directory")
    args = parser.parse_args()
    decoded_files = Path(args.outwavdir).glob("*.wav")

    for f in decoded_files:
        stem = str(f.stem).rstrip("_gen")
        org = stem.split("org")[1].split("cv")[0].lstrip("-").rstrip("_")
        (f.parent / org).mkdir(exist_ok=True, parents=True)
        f.rename((f.parent / org / (stem + ".wav")))


if __name__ == "__main__":
    main()
