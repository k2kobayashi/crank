#! /bin/bash
#
# download.sh
# Copyright (C) 2020 Wen-Chin HUANG
#
# Distributed under terms of the MIT license.
#

downloaddir=
voc=

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -eu # stop when error occured and undefined vars are used

case "${voc}" in
    "PWG") id="set id of google drive" ;;
    *) echo "No such pretrained model: ${voc}"; exit 1 ;;
esac

mkdir -p "${downloaddir}"
if [ ! -e "${downloaddir}"/.done ]; then
    gdown --id "$id" \
        --output "$downloaddir"/PWG.tar.gz
    tar zxvf "$downloaddir"/PWG.tar.gz \
        -C "$downloaddir"
    touch "${downloaddir}"/.done
fi
echo "Successfully finished donwload of pretrained model."
