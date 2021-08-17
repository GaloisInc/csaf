#!/bin/bash
PUBLISH=0
SCRIPT_DIR="csaf"

DIR=$(basename ${PWD})
if [ ${DIR} != ${SCRIPT_DIR} ]
then
    printf "ERROR: Script must be run from the \"${SCRIPT_DIR}\" directory!\n"
    exit 1
fi

source .common.sh

print_help() {
    printf "Usage: -p -t <tag_name>\n"
    printf "   -p      publish the image to docker hub\n"
    printf "   -t      the tag of the image { stable, edge, latest }\n"
    printf "   -h      prints the help menu\n"
    printf "\n"
}

publish_img() {

    show_info "Publishing image"
    docker push $IMAGE_NAME:$IMAGE_TAG

    if [[ ${?} -ne 0 ]]
    then
        show_error_and_exit "Unable to publish the image"
    fi
}

while getopts ":t:ph" opt; do
    case ${opt} in
    t )
        IMAGE_TAG=$OPTARG
        ;;
    p )
        PUBLISH=1
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

validate_tag "${IMAGE_TAG}"

build_img

if [[ ${PUBLISH} -eq 1 ]] ; then
    publish_img
fi
