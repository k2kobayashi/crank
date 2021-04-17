#! /bin/bash
#
# download.sh
# Copyright (C) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
#

downloaddir=downloads

# shellcheck disable=SC1091
. utils/parse_options.sh
set -e # stop when error occured

mkdir -p $downloaddir/wav
if [ ! -e "${downloaddir}"/.done ]; then
    echo "${downloaddir}/wav"
    touch downloads/.done
fi
