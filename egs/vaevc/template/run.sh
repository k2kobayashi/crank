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
conf=conf/mlfb_vqvae.yml # newtork config
spkr_yml=conf/spkr.yml   # speaker config

# synthesis related
voc=PWG                  # GL or PWG
voc_expdir=downloads/PWG # ex. `downloads/pwg`
voc_checkpoint=""        # If not specified, use the latest checkpoint

# other settings
resume_checkpoint="None" # checkpoint path to resume
decode_checkpoint="None" # checkpoint path to resume
dev_utterances=3  # # of development utterances
eval_utterances=5 # # of evaluation utterances
eval_speakers=""  # evaluation speaker

# parse options
. utils/parse_options.sh || exit 1;

set -eu # stop when error occured and undefined vars are used
[ $n_gpus -eq 0 ] && export CUDA_VISIBLE_DEVICES=""
# set decode step
feat_type=$(grep input_feat_type ${conf} | head -n 1 | awk '{print $2}')
if [ $decode_checkpoint != "None" ] ; then
    n_decode_steps=$(basename $decode_checkpoint | sed -e 's/[^0-9]//g')
else
    n_decode_steps=$(grep "n_steps:" "$conf" | awk '{print $2}')
fi

mkdir -p "${expdir}"
scpdir=${datadir}/scp
featdir=${datadir}/feature; mkdir -p ${featdir}
logdir=${datadir}/log; mkdir -p ${logdir}
confname=$(basename "${conf}" .yml)
featlabel=$(grep "label" < "${conf}" | head -n 1 | awk '{print $2}')

# stage 0: download dataset and generate scp
if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    echo "stage 0: download dataset and generate scp"
    # shellcheck disable=SC2154
    ${train_cmd} "${logdir}/download.log" \
        local/download.sh --downloaddir "${downloaddir}"
    ${train_cmd} "${logdir}/generate_scp.log" \
        python -m crank.bin.generate_scp \
            --wavdir "${downloaddir}"/wav \
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
            "${downloaddir}"/wav \
            "${datadir}/figure"
    echo "Please set speaker parameters in ${spkr_yml}"
    echo "stage 1: initialization has been done."
    exit
fi

# stage 2: feature extraction and calcualte statistics
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "stage 2: extract features and statistics"
    for phase in train dev eval; do
        ${train_cmd} "${featdir}/${featlabel}/extract_feature_${phase}.log" \
            python -m crank.bin.extract_feature \
                --n_jobs "${n_jobs}" \
                --phase "${phase}" \
                --conf "${conf}" \
                --spkr_yml "${spkr_yml}" \
                --scpdir "${scpdir}" \
                --featdir "${featdir}"
    done
    ${train_cmd} "${featdir}/${featlabel}/extract_statistics.log" \
        python -m crank.bin.extract_statistics \
            --n_jobs "${n_jobs}" \
            --phase train \
            --conf "${conf}" \
            --scpdir "${scpdir}" \
            --featdir "${featdir}"
    echo "stage 2: extract features and statistics has been done."
fi

# stage 3: model training
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "stage 3: train model"
    ${train_cmd} --gpu ${n_gpus} \
        "${expdir}/${confname}/train.log" \
        python -m crank.bin.train \
            --flag train \
            --n_jobs "${n_jobs}" \
            --conf "${conf}" \
            --checkpoint ${resume_checkpoint} \
            --scpdir "${scpdir}" \
            --featdir "${featdir}" \
            --expdir "${expdir}"
    echo "stage 3: train model has been done."
fi

# stage 4: reconstruction of training
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "stage 4: generate reconstruction"
    ${train_cmd} --gpu ${n_gpus} \
       "${expdir}/${confname}/reconstruction.log" \
        python -m crank.bin.train \
            --flag reconstruction \
            --n_jobs "${n_jobs}" \
            --conf "${conf}" \
            --checkpoint "${decode_checkpoint}" \
            --scpdir "${scpdir}" \
            --featdir "${featdir}" \
            --expdir "${expdir}"
    echo "stage 4: generate reconstruction has been done."
fi

# stage 5: decoding
if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    echo "stage 5: decode"
    ${train_cmd} --gpu ${n_gpus} \
       "${expdir}/${confname}/decode.log" \
        python -m crank.bin.train \
            --flag eval \
            --n_jobs "${n_jobs}" \
            --conf "${conf}" \
            --checkpoint "${decode_checkpoint}" \
            --scpdir "${scpdir}" \
            --featdir "${featdir}" \
            --featsscp "${featsscp}" \
            --expdir "${expdir}"
    echo "stage 5: decoding has been done."
fi

# stage 6: synthesis
if [ "${feat_type}" = "mcep" ]; then
    outdir=${expdir}/${confname}/eval_wav/${n_decode_steps}
else
    outdir=${expdir}/${confname}/eval_$(basename $voc_expdir)_wav/${n_decode_steps}
fi
if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    echo "stage 6: synthesis"
    mkdir -p "${outdir}/wav"
    # Griffin-Lim
    if [ ${voc} = "GL" ]; then
        echo "Griffin-Lim phase recovery"
        ${train_cmd} "${outdir}/griffin_lim_decode.log" \
            python -m crank.bin.griffin_lim \
                --n_jobs "${n_jobs}" \
                --conf "${conf}" \
                --rootdir ${expdir}/"${confname}"/eval_wav/"${n_decode_steps}" \
                --outdir "${outdir}/wav"

    # ParallelWaveGAN
    elif [ ${voc} = "PWG" ]; then
        echo "Parallel WaveGAN vocoder"
        mkdir -p "$voc_expdir"
        ${train_cmd} "${voc_expdir}/download_pretrained_vocoder.log" \
            local/download_pretrained_vocoder.sh \
                --downloaddir "$voc_expdir" \
                --voc ${voc}

        # variable settings
        [ -z "${voc_checkpoint}" ] && \
            voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 \
            | xargs -0 ls -t | head -n 1)"
        voc_conf="${voc_expdir}"/config.yml
        voc_stats="${voc_expdir}"/stats.h5

        # normalize and dump
        echo "Normalize decoded feature"
        mkdir -p "${outdir}"/hdf5_norm
        ${train_cmd} "${outdir}/normalize.log" \
            parallel-wavegan-normalize \
                --skip-wav-copy \
                --rootdir ${expdir}/"${confname}"/eval_wav/"${n_decode_steps}" \
                --config "${voc_conf}" \
                --stats "${voc_stats}" \
                --dumpdir "${outdir}"/hdf5_norm \
                --verbose 1
        echo "successfully finished normalization."

        # decoding
        echo "Decoding start. See the progress via ${outdir}/decode.log."
        ${train_cmd} --gpu ${n_gpus} "${outdir}/pwg_decode.log" \
            parallel-wavegan-decode \
                --dumpdir "${outdir}"/hdf5_norm \
                --checkpoint "${voc_checkpoint}" \
                --outdir "${outdir}"/wav \
                --verbose 1
        echo "successfully finished decoding."

        # rename
        echo "Rename decoded files "
        ${train_cmd} "${outdir}/rename_decoded.log" \
            python -m crank.bin.rename_decoded  \
                --outwavdir "${outdir}"/wav
        echo "successfully renamed decoded waveforms."
    else
        echo "Not supported decoder type. GL and PWG are available."
    fi
    echo "stage 6: synthesis has been done."
fi

# stage 7: evaluation
if [ "${stage}" -le 7 ] && [ "${stop_stage}" -ge 7 ]; then
    echo "stage 7: evaluation"
    echo "MCD calculation. Results: ${outdir}/mcd.log"
    ${train_cmd} "${outdir}/mcd.log" \
        python -m crank.bin.evaluate_mcd \
            --conf "${conf}" \
            --n_jobs "${n_jobs}" \
            --spkr_conf "${spkr_yml}" \
            --outwavdir "${outdir}/wav" \
            --featdir ${featdir}

    echo "MOSnet score prediction. Results: ${outdir}/mosnet.log"
    ${train_cmd} --gpu ${n_gpus} \
        "${outdir}/mosnet.log" \
        python -m crank.bin.evaluate_mosnet \
            --outwavdir "${outdir}/wav"
    echo "stage 7: evaluation has been done."
fi
