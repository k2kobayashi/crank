#! /bin/bash
#
# run.sh
# Copyright (C) 2020 Wen-Chin HUANG
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
#  5: evaluation
stage=0      # stage to start
stop_stage=5 # stage to stop

# job settings
n_jobs=10 # number of parallel jobs
n_gpus=1  # number of gpus

# directory settings
db_root=downloads
wavdir=downloads/wav # directory to save downloaded wav files
datadir=data         # directory to save list and features files
expdir=exp           # directory to save experiments
featsscp="None"

# config settings
conf=conf/default.yml  # newtork config
spkr_yml=conf/spkr.yml # speaker config

# other settings
checkpoint="None" # checkpoint path to resume
dev_utterances=10  # # of development utterances
eval_utterances=35 # # of evaluation utterances
eval_speakers=""  # evaluation speaker

# parse options
. utils/parse_options.sh || exit 1;

set -eu # stop when error occured and undefined vars are used

mkdir -p "${expdir}"
featdir=${datadir}/feature; mkdir -p ${featdir}
scpdir=${datadir}/scp; 
logdir=${datadir}/log; mkdir -p ${logdir}

# stage 0: download dataset and generate scp
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "stage 0: download dataset and generate scp"

    # download data
    local/download.sh ${db_root}
    
    # remake scpdir if exists
    [ -e ${scpdir} ] && rm -rf ${scpdir}
    mkdir -p ${scpdir}

    # generate scps
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
    echo "Please set speaker parametersn in ${spkr_yml}"
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

# stage 5: evaluation
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "stage 5: evaluate"
    confname=$(basename "${conf}" .yml)
    ${train_cmd} --gpu ${n_gpus} \
       "${expdir}/${confname}/evaluate.log" \
        python -m crank.bin.train \
            --flag eval \
            --n_jobs "${n_jobs}" \
            --conf "${conf}" \
            --checkpoint "${checkpoint}" \
            --scpdir "${scpdir}" \
            --featdir "${featdir}" \
            --featsscp "${featsscp}" \
            --expdir "${expdir}"
    echo "stage 5: evaluation has been done."
fi
