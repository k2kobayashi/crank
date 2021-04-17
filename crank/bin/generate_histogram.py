#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate histograms to decide speaker-dependent parameters
Lots of part copied from k2kobayashi/sprocket
https://github.com/k2kobayashi/sprocket/blob/master/LICENSE.txt
https://github.com/k2kobayashi/sprocket/blob/master/example/src/initialize_speaker.py

"""

import argparse
import os
import logging
from pathlib import Path
from joblib import Parallel, delayed

import matplotlib
import numpy as np
from scipy.io import wavfile

from sprocket.speech import FeatureExtractor
from crank.utils import low_cut_filter

matplotlib.use("Agg")  # noqa #isort:skip
import matplotlib.pyplot as plt  # noqa isort:skip

logging.basicConfig(level=logging.INFO)


def create_histogram(
    data, figure_path, range_min=-70, range_max=20, step=10, xlabel="Power [dB]"
):
    """Create histogram

    Parameters
    ----------
    data : list,
        List of several data sequences
    figure_path : str,
        Filepath to be output figure
    range_min : int, optional,
        Minimum range for histogram
        Default set to -70
    range_max : int, optional,
        Maximum range for histogram
        Default set to -20
    step : int, optional
        Stap size of label in horizontal axis
        Default set to 10
    xlabel : str, optional
        Label of the horizontal axis
        Default set to 'Power [dB]'

    """

    # plot histgram
    plt.hist(
        data,
        bins=200,
        range=(range_min, range_max),
        density=True,
        histtype="stepfilled",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xticks(np.arange(range_min, range_max, step))

    figure_dir = os.path.dirname(figure_path)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(figure_path)
    plt.close()


def extract_f0_and_npow(wavf):
    """
    F0 and npow extraction

    Parameters
    ----------
    wavf : str,
        File path of waveform file

    Returns
    -------
    dict :
        Dictionary consisting of F0 and npow arrays

    """

    # open waveform
    print("Extract: " + wavf)
    fs, x = wavfile.read(wavf)
    x = np.array(x, dtype=np.float32)
    x = low_cut_filter(x, fs, cutoff=70)

    # constract FeatureExtractor class
    feat = FeatureExtractor(analyzer="world", fs=fs)

    # f0 and npow extraction
    f0, _, _ = feat.analyze(x)
    npow = feat.npow()

    return {"f0": f0, "npow": npow}


def create_histograms(args, spkr):
    f0histogrampath = Path(args.figure_dir) / str(spkr + "_f0histogram.png")
    npowhistogrampath = Path(args.figure_dir) / str(spkr + "_npowhistogram.png")

    if not f0histogrampath.exists() and not npowhistogrampath.exists():
        logging.info("Histogram generation for {}".format(spkr))
        file_lists = (Path(args.wav_dir) / spkr).glob("*.wav")
        results = Parallel(n_jobs=args.n_jobs)(
            [delayed(extract_f0_and_npow)(str(f)) for f in file_lists]
        )

        # parse results
        f0s = [r["f0"] for r in results]
        npows = [r["npow"] for r in results]

        # stack feature vectors
        f0s = np.hstack(f0s).flatten()
        npows = np.hstack(npows).flatten()

        # create a histogram to visualize F0 range of the speaker
        create_histogram(
            f0s,
            f0histogrampath,
            range_min=40,
            range_max=700,
            step=50,
            xlabel="Fundamental frequency [Hz]",
        )

        # create a histogram to visualize npow range of the speaker
        create_histogram(
            npows,
            npowhistogrampath,
            range_min=-70,
            range_max=20,
            step=10,
            xlabel="Frame power [dB]",
        )


def main():
    dcp = "Create histogram for speaker-dependent configure"
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument("--n_jobs", type=int, default=-1, help="# of CPUs")
    parser.add_argument("--spkr", type=str, default=None, help="Label of a speaker")
    parser.add_argument("wav_dir", type=str, help="Directory of wav file")
    parser.add_argument("figure_dir", type=str, help="Directory for figure output")
    args = parser.parse_args()

    if args.spkr is not None:
        create_histograms(args, args.spkr)
    else:
        spkrs = [s.name for s in list(Path(args.wav_dir).iterdir())]
        for spkr in spkrs:
            create_histograms(args, spkr)


if __name__ == "__main__":
    main()
