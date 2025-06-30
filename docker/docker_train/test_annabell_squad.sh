#!/bin/bash

#script to test ANNABELL in specialist subjects
#     ./train_annabell_squad.sh
#    "data/statements/crossvalidation/round1/logs/logfile_squad_test.txt"
#    "data/statements/crossvalidation/round1/links/links_people_body_skills_squad.dat"
#    "data/statements/testing/all_statements.txt"

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

LOGFILE=$1
PRETRAINED_WEIGHTS=$2
TESTING_FILE=$3

(
#turn on logging
echo .logfile "$LOGFILE"
#load the weights
echo .load "$PRETRAINED_WEIGHTS"
#test using the questions and answers
echo .f "$TESTING_FILE"
#turn off logging
echo .logfile off
#shut down ANNABELL
echo .q
) | annabell
