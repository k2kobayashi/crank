#! /bin/bash
#
# download.sh
# Copyright (C) 2020 Wen-Chin HUANG
#
# Distributed under terms of the MIT license.
#

download_dir=
pretrained_model=

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -eu # stop when error occured and undefined vars are used

case "${pretrained_model}" in
    "PWG")             share_url="https://drive.google.com/open?id=1P7bwrEgDiqwGZCO9aEkFQ4YDd95qTbD0" ;;
    *) echo "No such pretrained model: ${pretrained_model}"; exit 1 ;;
esac

dir=${download_dir}
mkdir -p "${dir}"
if [ ! -e "${dir}"/.done ]; then
    utils/download_from_google_drive.sh "${share_url}" "${dir}" ".tar.gz"
    touch "${dir}"/.done
fi
echo "Successfully finished donwload of pretrained model."
