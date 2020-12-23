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
if [ ! -e "$downloaddir"/.done ]; then
    gdown \
        --id 19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt \
        --output "$downloaddir"/jsv_ver1.zip
    unzip "$downloaddir"/jsv_ver1.zip -d "$downloaddir"
    for tdir in "$downloaddir"/jvs_ver1/jvs* ; do
        spkr=$(basename "$tdir")
        ln -s \
            ../jvs_ver1/"$spkr"/parallel100/wav24kHz16bit \
            "$downloaddir"/wav/"$spkr"
    done
    touch "$downloaddir"/.done
else
    echo "dataset download already finished"
fi
