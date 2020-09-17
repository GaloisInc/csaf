#!/bin/bash
source .common.sh

validate_dir

docker run --network host -it -v $PWD:/${SCRIPT_DIR} $IMAGE_NAME:$IMAGE_TAG
