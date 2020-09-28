#!/bin/bash
IMAGE_NAME=galoisinc/csaf
IMAGE_TAG=latest
SCRIPT_DIR="csaf_architecture"

validate_dir() {

	DIR=$(basename ${PWD})
	if [ ${DIR} != ${SCRIPT_DIR} ]
	then
		printf "ERROR: Script must be run from the \"${SCRIPT_DIR}\" directory\n"
		exit 1
	fi
}

show_error_and_exit() {
	printf "ERROR: ${1}\n"
	exit 1
}

show_info() {
	printf "INFO: ${1}\n"
}

build_img() {

	show_info "Building image \"${IMAGE_NAME}:${IMAGE_TAG}\""
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

	if [[ ${?} -ne 0 ]]
	then
		show_error_and_exit "Unable to build image locally"
	fi
}

validate_tag() {

	if [ ${1} != "stable" ] && \
	   [ ${1} != "edge" ]  && \
	   [ ${1} != "latest" ]
	then
		show_error_and_exit "Image tag invalid"
	fi
}
