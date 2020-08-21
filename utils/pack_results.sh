#! /bin/bash
#
# pack_results.sh
# Copyright (C) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
#

# shellcheck disable=SC1091
. ./path.sh || exit 1

. utils/parse_options.sh

conf=$1

# check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <conf>"
    exit 1
fi

set -eu

# shellcheck disable=SC2012
recipe=$(basename `pwd`)
confname=$(basename "${conf}" .yml)
model_dir=exp/"${confname}"
PWG_dir=${model_dir}/eval_PWG_wav

ckpt=$(basename "$(ls -dt "${model_dir}"/*.pkl | head -1)")
wav_dir=$PWG_dir/$(basename "$(ls -dt "${PWG_dir}"/* | head -1)")/wav
tar_name="${recipe}_${confname}".tar.gz

tar -cvzf "${tar_name}" \
    "${conf}" \
    "${model_dir}/${ckpt}" "${wav_dir}"/mcd.log "${wav_dir}"/mosnet.log \
    "${wav_dir}"
