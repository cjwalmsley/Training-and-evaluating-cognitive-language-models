#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cjw81@kent.ac.uk
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --constraint=ampere
#SBATCH --mem=16G


#run with the following command: sbatch -p cpu --constraint=ampere --mem=16G ~/Training-and-evaluating-cognitive-language-models/slurm/slurm_run_pipline_prepared_database.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv_python3.13

python ~/Training-and-evaluating-cognitive-language-models/pipeline.py --use_prepared_dataset_if_available True