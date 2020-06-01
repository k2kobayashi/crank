#! /bin/bash
#
# download.sh
# Copyright (C) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.
#

wavdir=downloads/wav

# shellcheck disable=SC1091
. utils/parse_options.sh
set -e # stop when error occured

if [ ! -e downloads/.done ]; then
    echo "${wavdir}"
    touch downloads/.done
fi
