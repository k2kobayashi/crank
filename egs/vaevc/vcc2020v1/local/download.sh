#! /bin/bash
#
# download.sh
# Copyright (C) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
#

downloaddir=

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -eu # stop when error occured and undefined vars are used

wavdir=${downloaddir}/wav
[ ! -e ${downloaddir} ] && mkdir ${downloaddir}
[ ! -e "${wavdir}" ] && mkdir -p ${wavdir}

# TODO (k2koabayashi): Implement download
if [ ! -e downloads/.done ]; then
    echo "vcc2020 dataset has not released yet."
    echo "We will plan to implement the script after releasing."
    echo "If you have already had the dataset, you can put them "
    echo "like ``downloads/wav/{train,eval}/{SEF1,SEF2,...,TMM1}``."
    touch ${downloaddir}/.done
else
    echo "already finished. skipped download."
fi
