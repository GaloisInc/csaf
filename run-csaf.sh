#!/bin/bash
LOCAL=0
NATIVE=0
CSAF_LOC=""
CONFIG_NAME=""

source .common.sh

validate_dir

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
		show_error_and_exit "Unknown argument: " $OPTARG
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
	show_info "Building docker image locally."
	build_img
fi

if [[ ${NATIVE} -eq 1 ]] ; then
	if [[ ${JUYPTER} -eq 1 ]] ; then
		jupyter notebook --no-browser --notebook-dir=${PWD}/docs/notebooks
	else
		python3 "src/run_system.py" ${CSAF_LOC} ${CONFIG_NAME}
	fi
else
	if [[ ${JUYPTER} -eq 1 ]] ; then
		docker run --init -p 8888:8888 -it -v ${CSAF_LOC}:/csaf-system \
			-v ${PWD}/src:/app -v ${PWD}/docs/notebooks:/notebooks \
			${IMAGE_NAME}:${IMAGE_TAG} "jupyter" "notebook" "--port=8888" \
			"--no-browser" "--ip=0.0.0.0" "--allow-root" "--notebook-dir=/notebooks"
	else
		docker run --init -it -v ${PWD}/src:/app -v ${CSAF_LOC}:/csaf-system --network host \
			${IMAGE_NAME}:${IMAGE_TAG} python3 "/app/run_system.py" "/csaf-system" ${CONFIG_NAME}
	fi
fi

