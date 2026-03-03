enter the apptainer environment with the following command:
limactl shell apptainer
apptainer build --build-arg ANNABELL_VERSION=annabell annabell.sif annabell.def

#starts the annabell apptainer image in a shell and makes the apptainer directory available in the container for
read/write access
needs to be run from the directory where annabell.sif is located
apptainer shell -e --bind ./:/apptainer
/Users/chris/PycharmProjects/Training-and-evaluating-cognitive-language-models/apptainer/annabell.sif

Starts the aanabell apptainer and runs the script my_script.sh located in the apptainer directory

apptainer exec annabell.sif bash /app/pre_train_annabell_squad_nyc.sh test_logfile.txt test_training_file.txt
output_weights.dat