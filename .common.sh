#!/bin/bash
IMAGE_NAME=galoisinc/csaf
IMAGE_TAG=latest
SCRIPT_DIR="csaf_architecture"


# Color table
#     0 – Black
#     1 – Red
#     2 – Green
#     3 – Yellow
#     4 – Blue
#     5 – Magenta
#     6 – Cyan
#     7 – White


red=$(tput setaf 1)
blue=$(tput setaf 4)
normal=$(tput sgr0)

validate_dir() {

	DIR=$(basename ${PWD})
	if [ ${DIR} != ${SCRIPT_DIR} ]
	then
		printf "%40s\n" "${blue}WARNING: Script might not be running from the parent level \"${SCRIPT_DIR}\" directory. This might break paths. ${normal}"
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
