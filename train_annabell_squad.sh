#!/bin/bash

#assumes working directory has the following structure:
#---datasets
#  ---squad
#   ---new_york
#     ---train
#        ---q_a_s.tsv
#     ---validation
#       ---q_a_s.tsv
#---training
# ---nyc_statements.txt
# ---train_annabell.py
#---crossvalidation
# ---round1
#   ---links
#     --- links_people_body_skills.dat

#load the weights
(echo .load crossvalidation/round1/links/links_people_body_skills.dat
#train using the statements
echo .f training/nyc_statements.txt
#save the weights
echo .save crossvalidation/round1/links/links_people_body_skills_nyc.dat
#shut down ANNABELL
echo .q) | annabell
