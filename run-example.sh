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

EXAMPLE_NAME=$1

echo "${EXAMPLE_NAME}"

request_example() {
	show_error_and_exit "An example name is required [f16-shield, f16-simple, inv-pendulum]"
}

CONF_NAME=""

case $EXAMPLE_NAME in
	"f16-simple")
  		CONF_NAME="f16_simple_config.toml"
  		;;
	"f16-shield")
  		CONF_NAME="f16_shield_config.toml"
  		;;
	"inv-pendulum")
		CONF_NAME="inv_pendulum_config.toml"
		;;
	*)
		request_example
		;;
esac

./run-csaf.sh -l -d ${PWD}/examples -c ${CONF_NAME}
