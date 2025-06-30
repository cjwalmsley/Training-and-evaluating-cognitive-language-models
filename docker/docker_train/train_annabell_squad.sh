#!/bin/bash

#script to pretrain ANNABELL with language basics, then train ANNABELL in specialist subjects with declarative sentences
# ./generate_declaratative_sentences.sh "shared_data/statements/crossvalidation/round1/logs/logfile_squad_statements.txt" "shared_data/statements/crossvalidation/round1/links/links_people_body_skills.dat" "training/all_statements.txt" "shared_data/statements/crossvalidation/round1/links/links_people_body_skills_squad.dat"


#the script assumes working directory has the following structure:
#|---datasets
#!/bin/bash

#assumes working directory has the following structure:
# .
# ├── training
# │   └── all_statements.txt
# └── crossvalidation
#     └── round1
#         ├── links
#         │   └── links_people_body_skills.dat
#         └── logs

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <logfile> <pre-training_weights> <post-training_weights> <statements_file>"
    exit 1
fi

LOGFILE=$1
PRETRAINED_WEIGHTS=$2
POSTTRAINED_WEIGHTS=$3
STATEMENTS_FILE=$4

(
#turn on logging
echo .logfile "$LOGFILE"
#load the weights
echo .load "$PRETRAINED_WEIGHTS"
#train using the statements
echo .f "$STATEMENTS_FILE"
#save the weights
echo .save "$POSTTRAINED_WEIGHTS"
#turn off logging
echo .logfile off
#shut down ANNABELL
echo .q
) | annabell
