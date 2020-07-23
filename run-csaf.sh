#!/bin/bash

SCRIPT_DIR="csaf_architecture"

validate_dir() {

	DIR=$(basename ${PWD})      
	if [ ${DIR} != ${SCRIPT_DIR} ]
	then
		printf "ERROR: Script must be run from the \"${SCRIPT_DIR}\" directory\n"
		exit 1
	fi
}

validate_dir

source .common.sh

build_img

docker run --network host ${IMAGE_NAME}:${IMAGE_TAG}

#docker run --network host -it -v $PWD:/beat-pd-pdx -v $BEAT_PD_DIR:/datasets/beat-pd $IMAGE_NAME:$IMAGE_TAG
