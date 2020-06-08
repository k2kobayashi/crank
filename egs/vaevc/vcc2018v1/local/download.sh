#! /bin/bash
#
# download.sh
# Copyright (C) 2020 Wen-Chin HUANG
#
# Distributed under terms of the MIT license.
#

downloaddir=

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1
set -eu # stop when error occured and undefined vars are used

wavdir=wav
textdir=text
[ ! -e ${downloaddir}/${wavdir} ] && mkdir -p ${downloaddir}/${wavdir}
[ ! -e ${downloaddir}/${textdir} ] && mkdir -p ${downloaddir}/${textdir}

if [ ! -e ${downloaddir}/.done ]; then
    cd ${downloaddir}
    
    # download and decompress 
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_reference.zip
    wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation_transcriptions.tar.gz
    unzip -q vcc2018_database_training.zip
    unzip -q vcc2018_database_evaluation.zip
    unzip -q vcc2018_database_reference.zip
    tar xf vcc2018_database_evaluation_transcriptions.tar.gz

    # copy the wav
    for spk in vcc2018_training/VCC2*; do
        mkdir -p ${wavdir}/$(basename ${spk})
        cp ${spk}/*.wav ${wavdir}/$(basename ${spk})
    done
    for spk in vcc2018_evaluation/VCC2*; do
        cp ${spk}/*.wav ${wavdir}/$(basename ${spk})
    done
    for spk in vcc2018_reference/VCC2*; do
        cp ${spk}/*.wav ${wavdir}/$(basename ${spk})
    done
    
    # rename the folders
    cd ${wavdir}
    for d in */; do
        mv $d ${d//VCC2/}
    done
    cd ..
   
    # copy the transcriptions
    find vcc2018_training/Transcriptions -name "*.txt" -exec cp {} ${textdir} \;
    cp scripts/*.txt ${textdir}

    # remove compressed files and decompressed folders
    rm -f vcc2018_database_training.zip
    rm -f vcc2018_database_evaluation.zip
    rm -f vcc2018_database_reference.zip
    rm -f vcc2018_database_evaluation_transcriptions.tar.gz
    rm -rf vcc2018_training
    rm -rf vcc2018_evaluation
    rm -rf vcc2018_reference
    rm -rf scripts

    cd ..
    touch downloads/.done
else
    echo "Already finished. Skipping download."
fi
