#!/bin/bash

#script to load pretrained weights to give ANNABELL language basics, then train ANNABELL with declarative sentences
# ./train_annabell_squad.sh data/statements/pre-training/logfile_nyc_training.txt data/statements/pre-training/links_people_body_skills.dat data/statements/training/people_body_skills_nyc.dat data/statements/training/nyc_statements.txt


#the script assumes the working directory has the following structure:
# .
# ├── training
# │   └── nyc_statements.txt
# └── pre-training
#     └── annabell
#         └── crossvalidation
#             └── round1
#                 └── links
#                     └── links_people_body_skills.dat


if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <logfile> <pre-training_weights> <post-training_weights> <statements_file>"
    exit 1
fi

LOGFILE_BASE=$1
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
# Insert datetime before the file extension
LOGFILE="${LOGFILE_BASE%.*}_${DATETIME}.${LOGFILE_BASE##*.}"

PRETRAINED_WEIGHTS=$2
POSTTRAINED_WEIGHTS=$3
STATEMENTS_FILE=$4

{ time (
#turn on logging
echo .logfile "$LOGFILE"
#record the stats
echo .stat
#load the weights
echo .load "$PRETRAINED_WEIGHTS"
#train using the statements
echo .f "$STATEMENTS_FILE"
#save the weights
echo .save "$POSTTRAINED_WEIGHTS"
#record the stats
echo .stat
#turn off logging
echo .logfile off
#shut down ANNABELL
echo .q
) | annabell; } 2>> "$LOGFILE"
