#!/bin/bash 

# make docker to access all gpus
GPU_ACCESS="--gpus all"

# docker image name
DOCKER_IMAGE="wmh-prediction:0.2.2"

# CHANGEABLE: Location for the code's output "HOST:DOCKER"
# On the HOST's machine: /home/febrian/Downloads/output
# On the DOCKER's container: /home/docker/data
# Note: Please only change the HOST's machine location (not the DOCKER's)
OUTPUT_FOLDER="/home/febrian/WMH-prediction/dataset/output-test:/home/docker/data/IO/output"

# CHANGEABLE: Location of the MRI data that will be tested "HOST:DOCKER"
# On the HOST's machine: /home/febrian/LBC1936-data-test
# On the DOCKER's container: /home/docker/data/MRI
# Note: Please only change the HOST's machine location (not the DOCKER's)
DATA_FOLDER="/home/febrian/WMH-prediction/dataset:/home/docker/data/IO/input"

# run the python test-installation.py code
PYTHON_CODE="python test-installation.py"

docker run --rm $GPU_ACCESS -v $OUTPUT_FOLDER -v $DATA_FOLDER $DOCKER_IMAGE $PYTHON_CODE
