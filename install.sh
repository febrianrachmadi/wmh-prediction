#!/bin/bash 

# docker image name
DOCKER_IMAGE="wmh-prediction:0.2.2"

# Installation folder
INSTALLATION_FOLDER="../WMH-prediction/"

# This docker uses tensorflow: version 2.5.1-gpu
docker pull tensorflow/tensorflow:2.5.1-gpu
docker image build --force-rm -t $DOCKER_IMAGE $INSTALLATION_FOLDER
