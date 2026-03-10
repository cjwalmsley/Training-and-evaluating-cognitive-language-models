#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cjw81@kent.ac.uk
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --gres=cpu:ampere:1
#SBATCH --mem=16G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv_python3.13

python ~/Training-and-evaluating-cognitive-language-models/pipeline.py --use_prepared_dataset_if_available True