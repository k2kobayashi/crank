#! /bin/bash
#
# run.sh
# Copyright (C) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
#

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# stage settings
#  0: download dataset
#  1: initialization
#  2: feature extraction
#  3: training
#  4: reconstruction
#  5: decoding
#  6: synthesis
#  7: evaluation
stage=0      # stage to start
stop_stage=7 # stage to stop

# job settings
n_jobs=10 # number of parallel jobs
n_gpus=1  # number of gpus

# directory settings
downloaddir=downloads # directory to save downloaded wav files
datadir=data          # directory to save list and features files
expdir=exp            # directory to save experiments
featsscp="None"

# config settings
conf=conf/mlfb_vqvae.yml  # newtork config
spkr_yml=conf/spkr.yml # speaker config

# synthesis related
model_step=                     # If not specified, use the latest.
voc=PWG                         # GL or PWG
voc_expdir=downloads/PWG        # ex. `downloads/pwg`
voc_checkpoint=                 # If not specified, use the latest checkpoint

# other settings
checkpoint="None" # checkpoint path to resume
dev_utterances=3  # # of development utterances
eval_utterances=5 # # of evaluation utterances
eval_speakers=""  # evaluation speaker

# parse options
. utils/parse_options.sh || exit 1;

set -eu # stop when error occured and undefined vars are used

mkdir -p "${expdir}"
wavdir=${downloaddir}/wav
scpdir=${datadir}/scp
featdir=${datadir}/feature; mkdir -p ${featdir}
logdir=${datadir}/log; mkdir -p ${logdir}

# stage 0: download dataset and generate scp
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "stage 0: download dataset and generate scp"
    ${train_cmd} "${logdir}/download.log" \
        local/download.sh --downloaddir "${downloaddir}"
    ${train_cmd} "${logdir}/generate_scp.log" \
        python -m crank.bin.generate_scp \
            --wavdir "${wavdir}" \
            --scpdir "${scpdir}" \
            --spkr_yml "${spkr_yml}" \
            --dev_utterances "${dev_utterances}" \
            --eval_utterances "${eval_utterances}" \
            --eval_speakers "${eval_speakers}"
    echo "stage 0: download dataset and generate scp has been done."
fi

# stage 1: initialization
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "stage 1: initialization"
    ${train_cmd} "${logdir}/generate_histogram.log" \
        python -m crank.bin.generate_histogram \
            --n_jobs "${n_jobs}" \
            "${wavdir}" \
            "${datadir}/figure"
    echo "Please set speaker parameters in ${spkr_yml}"
    echo "stage 1: initialization has been done."
    exit
fi

# stage 2: feature extraction and calcualte statistics
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "stage 2: extract features and statistics"
    for phase in train dev eval; do
        ${train_cmd} "${logdir}/extract_feature_${phase}.log" \
            python -m crank.bin.extract_feature \
                --n_jobs "${n_jobs}" \
                --phase "${phase}" \
                --conf "${conf}" \
                --spkr_yml "${spkr_yml}" \
                --scpdir "${scpdir}" \
                --featdir "${featdir}"
    done
    ${train_cmd} "${logdir}/extract_statistics.log" \
        python -m crank.bin.extract_statistics \
            --n_jobs "${n_jobs}" \
            --phase train \
            --conf "${conf}" \
            --scpdir "${scpdir}" \
            --featdir "${featdir}" \
            --expdir "${expdir}"
    echo "stage 2: extract features and statistics has been done."
fi

# stage 3: model training
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "stage 3: train model"
    confname=$(basename "${conf}" .yml)
    ${train_cmd} --gpu ${n_gpus} \
        "${expdir}/${confname}/train.log" \
        python -m crank.bin.train \
            --flag train \
            --n_jobs "${n_jobs}" \
            --conf "${conf}" \
            --checkpoint ${checkpoint} \
            --scpdir "${scpdir}" \
            --featdir "${featdir}" \
            --expdir "${expdir}"
    echo "stage 3: train model has been done."
fi

# stage 4: reconstruction of training
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "stage 4: generate reconstruction"
    confname=$(basename "${conf}" .yml)
    ${train_cmd} --gpu ${n_gpus} \
       "${expdir}/${confname}/reconstruction.log" \
        python -m crank.bin.train \
            --flag reconstruction \
            --n_jobs "${n_jobs}" \
            --conf "${conf}" \
            --checkpoint "${checkpoint}" \
            --scpdir "${scpdir}" \
            --featdir "${featdir}" \
            --expdir "${expdir}"
    echo "stage 4: generate reconstruction has been done."
fi

# stage 5: decoding
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "stage 5: decode"
    confname=$(basename "${conf}" .yml)
    ${train_cmd} --gpu ${n_gpus} \
       "${expdir}/${confname}/decode.log" \
        python -m crank.bin.train \
            --flag eval \
            --n_jobs "${n_jobs}" \
            --conf "${conf}" \
            --checkpoint "${checkpoint}" \
            --scpdir "${scpdir}" \
            --featdir "${featdir}" \
            --featsscp "${featsscp}" \
            --expdir "${expdir}"
    echo "stage 5: decoding has been done."
fi

# stage 6: synthesis
confname=$(basename "${conf}" .yml)
[ -z "${model_step}" ] && model_step="$(find "${expdir}/${confname}" -name "*.pkl" -print0 \
    | xargs -0 ls -t | head -n 1 | cut -d"/" -f 3 | cut -d"_" -f 2 | cut -d"s" -f 1)"
outdir=${expdir}/${confname}/eval_${voc}_wav/${model_step}
outwavdir=${outdir}/wav
if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    echo "stage 6: synthesis"
    mkdir -p "${outwavdir}"

    # Gliffin-Lim
    if [ ${voc} = "GL" ]; then
        echo "Using Griffin-Lim phase recovery."
        ${train_cmd} "${outwavdir}/decode.log" \
            python -m crank.bin.griffin_lim \
                --conf "${conf}" \
                --rootdir ${expdir}/"${confname}"/eval_wav/"${model_step}" \
                --outdir "${outwavdir}"

    # ParallelWaveGAN
    elif [ ${voc} = "PWG" ]; then
        echo "Using Parallel WaveGAN vocoder."
        if [ ! -d ${voc_expdir} ]; then
            echo "Downloading pretrained PWG model..."
            local/pretrained_model_download.sh \
                --download_dir ${voc_expdir} \
                --pretrained_model ${voc}
        fi
        echo "PWG model exists: ${voc_expdir}"

        # variable settings
        [ -z "${voc_checkpoint}" ] && voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 \
            | xargs -0 ls -t | head -n 1)"
        voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
        voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
        hdf5_norm_dir=${outdir}/hdf5_norm; mkdir -p "${hdf5_norm_dir}"

        # normalize and dump
        echo "Normalizing..."
        ${train_cmd} "${hdf5_norm_dir}/normalize.log" \
            parallel-wavegan-normalize \
                --skip-wav-copy \
                --rootdir ${expdir}/"${confname}"/eval_wav/"${model_step}" \
                --config "${voc_conf}" \
                --stats "${voc_stats}" \
                --dumpdir "${hdf5_norm_dir}" \
                --verbose 1
        echo "successfully finished normalization."

        # decoding
        echo "Decoding start. See the progress via ${outwavdir}/decode.log."
        ${cuda_cmd} --gpu 1 "${outwavdir}/decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${hdf5_norm_dir}" \
                --checkpoint "${voc_checkpoint}" \
                --outdir "${outwavdir}" \
                --verbose 1
        echo "successfully finished decoding."

        # rename
        find "${outwavdir}" -name '*.wav' | sed -e "p;s/_gen//" | xargs -n2 mv
    else
        echo "Vocoder type not supported. Only GL and PWG are available."
    fi
    echo "stage 6: synthesis has been done."
fi

# stage 7: evaluation
if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
    echo "stage 7: evaluation"

    echo "MCD calculation. Results can be found in ${outwavdir}/mcd.log"
    feat_type=$(grep feat_type ${conf} | head -n 1 | awk '{ print $2}')
    if [ "${feat_type}" = "mcep" ]; then
        outwavdir=${expdir}/${confname}/eval_wav/${model_step}
    fi
    ${train_cmd} "${outwavdir}/mcd.log" \
        python -m crank.bin.evaluate_mcd \
            --conf "${conf}" \
            --spkr_conf "${spkr_yml}" \
            --outwavdir "${outwavdir}" \
            --featdir ${featdir}

    echo "MOSnet score prediction. Results can be found in ${outwavdir}/mosnet.log"
    ${train_cmd} --gpu 1 \
        "${outwavdir}/mosnet.log" \
        python -m crank.bin.evaluate_mosnet \
            --outwavdir "${outwavdir}"
    echo "stage 7: evaluation has been done."
fi
