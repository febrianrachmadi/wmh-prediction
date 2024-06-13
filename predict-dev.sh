#!/bin/bash 

# make docker to access all gpus
GPU_ACCESS="--gpus all"

# docker image name
DOCKER_IMAGE="wmh-prediction:0.2.2"

# CHANGEABLE: Location of the MRI data that will be tested "DATA_HOST:DOCKER"
# On the HOST's machine (DATA_HOST): /home/febrian/LBC1936-data-test
# On the DOCKER's container: /home/docker/data/MRI
# Note: Please only change the HOST's machine location (not the DOCKER's)
HOME_HOST="/home/febrian/WMH-prediction"
DOCKER_IO_DIR="/wmh-prediction/data/IO"
DATA_HOST=$HOME_HOST"/dataset"
DATA_DOCKER=$DOCKER_IO_DIR"/input"
DATA_VOLUME=$DATA_HOST":"$DATA_DOCKER

# Folders for T2 FLAIR brain MRI scans and their corresponding stroke lesions masks. For examples:
# L001.nii.gz
# L002.nii.gz
# L003.nii.gz
T2_FLAIR_BRAIN_DIR=$DATA_DOCKER"/study_1_brain_flair"

# Folders for the corresponding stroke lesions masks. Skip if there are no stroke lesions. For examples:
# L003.nii.gz --> i.e., only data L003 has corresponding stroke lesions
STROKE_MASK_DIR=$DATA_DOCKER"/study_1_stroke"

# CHANGEABLE: Location for the code's output "HOST:DOCKER"
# On the HOST's machine: $DATA_DOCKER"/study_1_output" (please only change the location inside the "")
# On the DOCKER's container: /home/docker/data
# Note: Please only change the HOST's machine location (not the DOCKER's)
OUTPUT_VOLUME_DIR=$DATA_DOCKER"/study_1_output"
OUTPUT_VOLUME=$OUTPUT_VOLUME_DIR":"$DOCKER_IO_DIR"/output"

# CSV file containing list of T2 FLAIR brain MRI files inside T2_FLAIR_BRAIN_DIR. For examples:
# L001.nii.gz
# L002.nii.gz
# L003.nii.gz
STUDY_DATA_LIST=$DATA_DOCKER"/study_data_list.csv"

# Number of probabilistic iterations per model. Recommendations:
# 1 (faster)
# 5 (moderate)
# 10 (slow)
# 30 (slower)
NUM_INFERENCE_ITERATIONS=5

# Folders for developments
DEV_HOST=$HOME_HOST"/dev"
DEV_DOCKER=$DOCKER_IO_DIR"/dev"
DEV_VOLUME=$DEV_HOST":"$DEV_DOCKER

# run the python test-installation.py code
PYTHON_CODE=$DEV_DOCKER"/inference-dev.py"
PYTHON_CMD="python "$PYTHON_CODE

docker run --rm -it $GPU_ACCESS -v $OUTPUT_VOLUME -v $DATA_VOLUME -v $DEV_VOLUME $DOCKER_IMAGE $PYTHON_CMD $T2_FLAIR_BRAIN_DIR $STROKE_MASK_DIR $STUDY_DATA_LIST $OUTPUT_VOLUME_DIR $NUM_INFERENCE_ITERATIONS
