#!/bin/bash

#script to test ANNABELL in specialist subjects
#     ./test_annabell_squad.sh
#    "data/statements/testing/test_nyc_log.txt"
#    "data/statements/testing/training_nyc_weights"
#    "data/statements/testing/test_nyc_questions.txt"

#the script assumes working directory has the following structure:
#|---datasets
#!/bin/bash

#assumes working directory has the following structure:
# .
# ├── testing
# │   └── test_all_questions.txt
# └── crossvalidation
#     └── round1
#         ├── links
#         │   └── links_people_body_skills_squad.dat
#         └── logs

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <logfile> <pre-training_weights> <testing_file>"
    exit 1
fi

LOGFILE_BASE=$1
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)
# Insert datetime before the file extension
LOGFILE="${LOGFILE_BASE%.*}_${DATETIME}.${LOGFILE_BASE##*.}"
PRETRAINED_WEIGHTS=$2
TESTING_FILE=$3

# The time command's output (stderr) is appended to the log file.
{ time (
    #turn on logging
    echo .logfile "$LOGFILE"
    #record the stats
    echo .stat
    #load the weights
    echo .load "$PRETRAINED_WEIGHTS"
    #test using the questions and answers
    echo .f "$TESTING_FILE"
    #record the stats
    echo .stat
    #turn off logging
    echo .logfile off
    #shut down ANNABELL
    echo .q
) | annabell; } 2>> "$LOGFILE"
