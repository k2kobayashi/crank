#! /bin/bash
#
# generate_results.sh
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
voc_confname=$(basename "${voc_expdir}")
PWG_dir=exp/${confname}/eval_${voc_confname}_wav
# shellcheck disable=SC2012
result_dir=$PWG_dir/$(basename "$(ls -dt "${PWG_dir}"/* | head -1)")

mcd=$(head -n -2 < "$result_dir"/mcd.log | tac | head -n -4 | tac | awk '{if($1!=$2) print $1, $2, $3}' | awk '{sum+=$3} END {print sum/NR}')
mosnet=$(head -n -2 "$result_dir"/mosnet.log | tac | head -n -8 | tac | awk '{if($1!=$2) print $1, $2, $3}' | awk '{sum+=$3} END {print sum/NR}')
echo "# $recipe: $confname.conf"
echo "- mcd"
echo "  - $mcd"
echo "- mosnet"
echo "  - $mosnet"
echo "---"
