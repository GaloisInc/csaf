#!/bin/bash

SCRIPT_DIR="csaf_architecture"
EXAMPLE_NAME=""
NATIVE=""
LOCAL=""
CONF_NAME=""
EXAMPLE_DIR=""

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
	printf "   -e      the name of the example { f16-shield, f16-simple, inv-pendulum, rpc-inv-pendulum }\n"
	printf "   -l      build the image locally\n"
	printf "   -n      run CSAF natively\n"
	printf "   -h      prints the help menu\n"
	printf "\n"
}

while getopts ":e:lnh" opt; do
	case ${opt} in
        e )
		EXAMPLE_NAME=$OPTARG
		;;
        l )
		LOCAL="-l"
		;;
        n )
		NATIVE="-n"
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

if [[ ! -z ${LOCAL} && ! -z ${NATIVE} ]] ; then
	show_error_and_exit "the \'native\' and \'local\' options cannot be combined"
fi

if [ -z ${EXAMPLE_NAME} ]
then
	show_error_and_exit "An example name is required [f16-shield, f16-simple, inv-pendulum, rpc-inv-pendulum]"
fi

case $EXAMPLE_NAME in
	"f16-simple")
  		CONF_NAME="f16_simple_config.toml"
		EXAMPLE_DIR="examples/f16"
  		;;
	"f16-shield")
  		CONF_NAME="f16_shield_config.toml"
		EXAMPLE_DIR="examples/f16"
  		;;
	"inv-pendulum")
		CONF_NAME="inv_pendulum_config.toml"
		EXAMPLE_DIR="examples/inverted-pendulum"
		;;
	"rpc-inv-pendulum")
		CONF_NAME="rpc_inv_pendulum_config.toml"
		EXAMPLE_DIR="examples/rpc-inverted-pendulum"
		;;
	*)
		request_example
		;;
esac

./run-csaf.sh ${LOCAL} ${NATIVE} -d "${PWD}/${EXAMPLE_DIR}" -c ${CONF_NAME}
