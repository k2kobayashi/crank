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

mkdir -p downloads/wav
if [ ! -e downloads/.done ]; then
    gdown \
        --id 19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt \
        --output downloads/jsv_ver1.zip
    unzip downloads/jsv_ver1.zip -d downloads
    for tdir in downloads/jvs_ver1/jvs* ; do
        spkr=$(basename "$tdir")
        ln -s \
            ../jvs_ver1/"$spkr"/parallel100/wav24kHz16bit \
            downloads/wav/"$spkr"
    done
    touch downloads/.done
fi
