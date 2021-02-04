#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Utilities

"""

import logging
from distutils.version import LooseVersion
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.interpolate import interp1d
from scipy.signal import firwin, lfilter
from sprocket.speech import Synthesizer
from sprocket.util import HDF5

mpl.use("Agg")  # noqa


def open_featsscp(featsscp):
    feats = {}
    with open(featsscp) as fp:
        for line in fp.readlines():
            uid, h5f = line.rstrip().split()
            feats[uid] = h5f
    return feats


def open_scpdir(scpdir):
    scp = {"wav": {}, "feats": {}, "utt2spk": {}, "spk2utt": {}}

    scpdir = Path(scpdir)
    with open(scpdir / "wav.scp") as fp:
        for line in fp.readlines():
            uid, wavf = line.rstrip().split()
            scp["wav"][uid] = wavf

    with open(scpdir / "utt2spk") as fp:
        for line in fp.readlines():
            uid, spkr = line.rstrip().split()
            scp["utt2spk"][uid] = spkr

    with open(scpdir / "spk2utt") as fp:
        spkrs = []
        for line in fp.readlines():
            line = line.rstrip().split()
            spkrs.append(line[0])
            scp["spk2utt"][line[0]] = line[1:]
    scp["spkrs"] = spkrs

    return scp


def load_yaml(ymlf):
    with open(ymlf) as fp:
        return yaml.load(fp, Loader=yaml.SafeLoader)


def plot_mlfb(mlfb, path, ext="png"):
    plt.figure()
    plt.imshow(mlfb.T, origin="lower")
    plt.savefig(str(path) + "." + ext)
    plt.close()


def mlfb2wav(
    mlfb,
    fs=22050,
    n_mels=80,
    fftl=1024,
    win_length=1024,
    hop_size=220,
    fmin=80,
    fmax=7600,
    window="hann",
    n_iters=100,
):
    spc = logmelspc_to_linearspc(mlfb, fs, n_mels, fftl, fmin=fmin, fmax=fmax)
    return griffin_lim(spc, fftl, hop_size, win_length, window=window, n_iters=n_iters)


def mlfb2wavf(
    mlfb,
    wavf,
    fs=22050,
    n_mels=80,
    fftl=1024,
    win_length=1024,
    hop_size=220,
    fmin=80,
    fmax=7600,
    window="hann",
    n_iters=100,
    plot=False,
):
    Path(wavf).parent.mkdir(parents=True, exist_ok=True)
    try:
        wav = mlfb2wav(
            mlfb,
            fs=fs,
            n_mels=n_mels,
            fftl=fftl,
            win_length=win_length,
            hop_size=hop_size,
            fmin=fmin,
            fmax=fmax,
            window=window,
            n_iters=n_iters,
        )
        sf.write(str(wavf), wav, fs)
    except librosa.util.exceptions.ParameterError:
        logging.info("ERROR: GliffinLim for {}".format(str(wavf)))

    if plot:
        plot_mlfb(mlfb, wavf)


def feat2hdf5(mlfb, hdf5, ext="feats"):
    tdir, name = Path(hdf5).parent, Path(hdf5).stem
    h5f = tdir / (str(name) + ".h5")
    h5 = HDF5(str(h5f), "a")
    h5.save(mlfb, ext=ext)
    h5.close()


def world2wav(
    f0,
    mcep,
    codeap,
    rmcep=None,
    wavf=None,
    fs=22050,
    fftl=1024,
    shiftms=10,
    alpha=0.455,
):
    synthesizer = Synthesizer(fs=fs, fftl=fftl, shiftms=shiftms)
    wav = synthesizer.synthesis(f0, mcep, codeap, rmcep=rmcep, alpha=alpha)
    wav = np.clip(wav, -1.0, 1.0)
    if wavf is not None:
        sf.write(wavf, wav, fs)
    else:
        return wav


def diff2wav(
    x, diffmcep, rmcep, wavf=None, fs=22050, fftl=1024, shiftms=10, alpha=0.455
):
    synthesizer = Synthesizer(fs=fs, fftl=fftl, shiftms=shiftms)
    wav = synthesizer.synthesis_diff(x, diffmcep, rmcep=rmcep, alpha=alpha)
    wav = np.clip(wav, -1.0, 1.0)
    if wavf is not None:
        sf.write(wavf, wav, fs)
    else:
        return wav


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def low_cut_filter(x, fs, cutoff=70):
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def to_device(batch, device):
    for k, v in batch.items():
        if k == "h_scalar":
            batch[k] = v.type(torch.LongTensor)
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def logmelspc_to_linearspc(lmspc, fs, n_mels, n_fft, fmin=None, fmax=None):
    """Convert log Mel filterbank to linear spectrogram.
    This function is originally implemented in espnet/espnet
    https://github.com/espnet/espnet/blob/master/utils/convert_fbank_to_wav.py
    https://github.com/espnet/espnet/blob/master/LICENSE

    Args:
        lmspc (ndarray): Log Mel filterbank (T, n_mels).
        fs (int): Sampling frequency.
        n_mels (int): Number of mel basis.
        n_fft (int): Number of FFT points.
        f_min (int, optional): Minimum frequency to analyze.
        f_max (int, optional): Maximum frequency to analyze.
    Returns:
        ndarray: Linear spectrogram (T, n_fft // 2 + 1).
    """
    assert lmspc.shape[1] == n_mels
    EPS = 1e-10
    print(fs, n_mels, n_fft, fmin)
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    mspc = np.power(10.0, lmspc)
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    # spc = np.maximum(EPS, np.matmul(inv_mel_basis, mspc.T).T)
    spc = np.matmul(inv_mel_basis, mspc.T).T
    return spc


def griffin_lim(spc, n_fft, n_shift, win_length, window="hann", n_iters=100):
    """Convert linear spectrogram into waveform using Griffin-Lim.
    This function is originally implemented in espnet/espnet
    https://github.com/espnet/espnet/blob/master/utils/convert_fbank_to_wav.py
    https://github.com/espnet/espnet/blob/master/LICENSE

    Args:
        spc (ndarray): Linear spectrogram (T, n_fft // 2 + 1).
        n_fft (int): Number of FFT points.
        n_shift (int): Shift size in points.
        win_length (int): Window length in points.
        window (str, optional): Window function type.
        n_iters (int, optionl): Number of iterations of Griffin-Lim Algorithm.
    Returns:
        ndarray: Reconstructed waveform (N,).
    """
    # assert the size of input linear spectrogram
    assert spc.shape[1] == n_fft // 2 + 1

    # use librosa's fast Grriffin-Lim algorithm
    spc = np.abs(spc.T)
    y = np.clip(
        librosa.core.griffinlim(
            S=spc,
            n_iter=n_iters,
            hop_length=n_shift,
            win_length=win_length,
            window=window,
        ),
        -1,
        0.999969482421875,
    )
    return y


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0
    This function is originally implemented in kan-bayashi/PytorchWaveNetVocoder
    https://github.com/kan-bayashi/PytorchWaveNetVocoder/blob/master/wavenet_vocoder/bin/feature_extract.py
    https://github.com/kan-bayashi/PytorchWaveNetVocoder/blob/master/LICENSE

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0
