#! /bin/bash
#
# download.sh
# Copyright (C) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
#

wavdir=downloads/wav

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -eu # stop when error occured and undefined vars are used

[ ! -e downloads ] && mkdir downloads
[ ! -e "${wavdir}" ] && mkdir -p ${wavdir}

# TODO (k2koabayashi): Implement download
if [ ! -e downloads/.done ]; then
    echo "vcc2020 dataset has not released yet."
    echo "We will plan to implement the script after releasing."
    echo "If you have already had the dataset, you can put them "
    echo "like ``downloads/wav/{train,eval}/{SEF1,SEF2,...,TMM1}``."
    touch downloads/.done
else
    echo "already finished. skipped download."
fi
