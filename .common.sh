#!/bin/bash
IMAGE_NAME=galoisinc/csaf
IMAGE_TAG=latest

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
