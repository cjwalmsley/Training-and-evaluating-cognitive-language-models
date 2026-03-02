enter the apptainer environment with the following command:
limactl shell apptainer
apptainer build --build-arg ANNABELL_VERSION=annabell annabell.sif annabell.def

#starts annabell in a shell and makes the apptainer directory available in the container for read/write access
needs to be run from the directory where annabell.sif is located
apptainer shell -e --bind ./:/apptainer
/Users/chris/PycharmProjects/Training-and-evaluating-cognitive-language-models/apptainer/annabell.sif

#copy the sif to the cluster
rsync -P ./annabell.sif cjw81@hydra.kent.ac.uk:

#copy back to local machine (example)
rsync -P cjw81@hydra.kent.ac.uk:~/Training-and-evaluating-cognitive-language-models/slurm/annabell.sif .
