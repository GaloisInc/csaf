#!/bin/bash
SCRIPT_DIR="csaf_architecture"
DIR=$(basename ${PWD})
if [ ${DIR} != ${SCRIPT_DIR} ]
then
	printf "ERROR: Script must be run from the \"${SCRIPT_DIR}\" directory!\n"
	exit 1
fi

source .common.sh

docker run --network host -it -v $PWD:/${SCRIPT_DIR} $IMAGE_NAME:$IMAGE_TAG
