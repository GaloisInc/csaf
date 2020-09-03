#!/bin/bash

SCRIPT_DIR="csaf_architecture"
LOCAL=0
NATIVE=0
CSAF_LOC=""
CONFIG_NAME=""

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
	printf "   -c      the name of the config file\n"
	printf "   -d      fully qualified path to the directory defining the control system\n"
	printf "   -j      launch a jupyter notebook\n"
	printf "   -l      build the image locally\n"
	printf "   -n      run CSAF natively\n"
	printf "   -t      the tag of the image { stable, edge, latest }\n"
	printf "   -h      prints the help menu\n"
	printf "\n"
}

while getopts ":c:d:t:jlhn" opt; do
	case ${opt} in
        c )
		CONFIG_NAME=$OPTARG
		;;
        d )
		CSAF_LOC=$OPTARG
		;;
        j )
		JUYPTER=1
		;;
        l )
		LOCAL=1
		;;
        n )
		NATIVE=1
		;;
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

if [ -z ${CONFIG_NAME} ]
then
	show_error_and_exit "A CSAF config file must be supplied"
fi

if [ -z ${CSAF_LOC} ]
then
	show_error_and_exit "A CSAF directory must be supplied"
fi

if [[ ${LOCAL} -eq 1 && ${NATIVE} -eq 1 ]] ; then
	show_error_and_exit "the \'native\' and \'local\' options cannot be combined"
fi

if [[ ${JUYPTER} -eq 1 && ${NATIVE} -eq 1  ]] ; then
	show_error_and_exit "the \'native\' and \'jupyter\' options cannot be combined"
fi

if [[ ${LOCAL} -eq 1 ]] ; then
	build_img
else
	show_info "Pulling image from Docker Hub"
	# TODO
fi

if [[ ${NATIVE} -eq 1 ]] ; then
	python3 "src/run_system.py" ${CSAF_LOC} ${CONFIG_NAME}
else
	if [[ ${JUYPTER} -eq 1 ]] ; then
		docker run -p 8888:8888 -it -v ${CSAF_LOC}:/csaf-system ${IMAGE_NAME}:${IMAGE_TAG} "jupyter" "notebook" "--port=8888" "--no-browser" "--ip=0.0.0.0" "--allow-root" "--notebook-dir=/notebooks";
	else
		docker run -it -v ${CSAF_LOC}:/csaf-system --network host ${IMAGE_NAME}:${IMAGE_TAG} python3 "/app/run_system.py" "/csaf-system" ${CONFIG_NAME}
	fi
fi
