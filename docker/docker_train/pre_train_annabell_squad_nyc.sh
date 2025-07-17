#!/bin/bash

#script to pretrain ANNABELL with language basics using the samples from the NYC subset of the SQuAD dataset
# ./pre_train_annabell_squad_nyc.sh "shared_data/statements/crossvalidation/round1/logs/logfile_pretraining_nyc.txt" "training/pre_training_nyc" "shared_data/statements/crossvalidation/round1/links/links_pretraining_nyc_squad.dat"

#the script assumes working directory has the following structure:
#|---datasets

#assumes working directory has the following structure:
# .
# ├── training
# │   └── pre_training_nyc.txt
# └── crossvalidation
#     └── round1
#         └── logs

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <logfile> <training_file> <pre-trained_weights>"
    exit 1
fi

LOGFILE=$1
TRAINING_FILE=$2
PRETRAINED_WEIGHTS=$3

(
#turn on logging
echo .logfile "$LOGFILE"
#train using the commands provided in the file
echo .f "$TRAINING_FILE"
#save the weights
echo .save "$PRETRAINED_WEIGHTS"
#record the stats
echo .stat
#turn off logging
echo .logfile off
#shut down ANNABELL
echo .q
) | annabell
