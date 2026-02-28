#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv_python3.13

# Define variables for the arguments, using provided parameters or defaults
MODEL_STRING="${1:-llama3.2}"
TITLE="${2:-Frédéric_Chopin}"
NUMBER_OF_SENTENCES="${3:-5}" # Note: The python script expects a string for this argument if it's not 'all'

echo "Running with MODEL_STRING: $MODEL_STRING"
echo "Running with TITLE: $TITLE"
echo "Running with NUMBER_OF_SENTENCES: $NUMBER_OF_SENTENCES"

# Execute the Python script
python ~/Training-and-evaluating-cognitive-language-models/generate_declarative_sentences.py "$MODEL_STRING" "$TITLE" "$NUMBER_OF_SENTENCES"

echo "Script execution finished."