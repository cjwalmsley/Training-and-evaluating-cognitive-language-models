#!/bin/bash

#script to load pretrained weights, then train ANNABELL with additional data

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <logfile> <pre-training_weights> <post-training_weights> <statements_file>"
    exit 1
fi

LOGFILE=$1
PRETRAINED_WEIGHTS=$2
POSTTRAINED_WEIGHTS=$3
STATEMENTS_FILE=$4

# The time command's output (stderr) is appended to the log file.
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