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

print_help() {
	printf "Usage: -t <tag_name>\n"
	printf "   -t      the tag of the image { stable, edge, latest }\n"
	printf "   -h      prints the help menu\n"
	printf "\n"
}

while getopts ":t:h" opt; do
	case ${opt} in
	t )
		IMAGE_TAG=$OPTARG
		;;
	h )
		print_help
		exit 0
		;;
	* )
		show_error_and_exit "Unknown argument"
		;;
	esac
done

build_img

docker run --network host ${IMAGE_NAME}:${IMAGE_TAG}

# TODO
#docker run --network host -it -v $PWD:/beat-pd-pdx -v $BEAT_PD_DIR:/datasets/beat-pd $IMAGE_NAME:$IMAGE_TAG
