#!/bin/bash

# make docker to access all gpus
GPU_ACCESS="--gpus all"

# docker image name
DOCKER_IMAGE="wmh-prediction:0.2.2"

HOME_HOST="./wmh-prediction"
DOCKER_IO_DIR="/wmh-prediction/data/IO"

# CHANGEABLE: Location for the code's output "HOST:DOCKER"
# On the HOST's machine: /home/febrian/Downloads/output
# On the DOCKER's container: /home/docker/data
# Note: Please only change the HOST's machine location (not the DOCKER's)
OUTPUT_FOLDER=$HOME_HOST"/dataset/output-bash:"$DOCKER_IO_DIR"/output"

# CHANGEABLE: Location of the MRI data that will be tested "HOST:DOCKER"
# On the HOST's machine: /home/febrian/LBC1936-data-test
# On the DOCKER's container: /home/docker/data/MRI
# Note: Please only change the HOST's machine location (not the DOCKER's)
DATA_FOLDER=$HOME_HOST"/dataset:"$DOCKER_IO_DIR"/input"

# run the python test-installation.py code
DEV_HOST=$HOME_HOST"/dev"
DEV_DOCKER="/wmh-prediction/dev"
DEV_VOLUME=$DEV_HOST":"$DEV_DOCKER

docker run -u $(id -u):$(id -g) -it $GPU_ACCESS -v $OUTPUT_FOLDER -v $DATA_FOLDER -v $DEV_VOLUME $DOCKER_IMAGE bash
