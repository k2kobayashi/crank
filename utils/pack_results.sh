#! /bin/bash
#
# pack_results.sh
# Copyright (C) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
#

# shellcheck disable=SC1091
. ./path.sh || exit 1

. utils/parse_options.sh || exit 1;

conf=$1
voc_expdir=$2
# check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <conf>"
    exit 1
fi

set -eu

# shellcheck disable=SC2012
recipe=$(basename "$PWD")
confname=$(basename "${conf}" .yml)
model_dir=exp/"${confname}"
voc_confname=$(basename "${voc_expdir}")
PWG_dir=${model_dir}/eval_${voc_confname}_wav

# shellcheck disable=SC2012
ckpt=$(basename "$(ls -dt "${model_dir}"/*.pkl | head -1)")
# shellcheck disable=SC2012
wav_dir=$PWG_dir/$(basename "$(ls -dt "${PWG_dir}"/* | head -1)")/wav
tar_name="${recipe}_${confname}_${voc_confname}".tar.gz

tar -cvzf "${tar_name}" \
    "${conf}" \
    "${model_dir}/${ckpt}" "${wav_dir}"/mcd.log "${wav_dir}"/mosnet.log \
    "${wav_dir}"
