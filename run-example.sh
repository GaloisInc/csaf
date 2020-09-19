#!/bin/bash
EXAMPLE_NAME=""
JUPYTER=""
NATIVE=""
LOCAL=""
CONF_NAME=""
EXAMPLE_DIR=""
JOB_CONFIG_PATH=""

source .common.sh

validate_dir

print_help() {
	printf "Usage:\n"
	printf "   -e      the name of the example { f16-shield, f16-simple, inv-pendulum }\n"
	printf "   -f      fully qualified path to the job config file\n"
	printf "   -j      launch a jupyter notebook\n"
	printf "   -l      build the image locally\n"
	printf "   -n      run CSAF natively\n"
	printf "   -t      <tag_name>\n"
	printf "   -h      prints the help menu\n"
	printf "\n"
}

while getopts ":e:jf:lnh" opt; do
	case ${opt} in
        e )
		EXAMPLE_NAME=$OPTARG
		;;
		f )
		JOB_CONFIG_PATH=$OPTARG
		;;
        j )
		JUPYTER="-j"
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
		show_error_and_exit "Unknown argument: " $OPTARG
		;;
	esac
done

if [ -z ${EXAMPLE_NAME} ]
then
	show_error_and_exit "An example name is required [f16-shield, f16-simple, inv-pendulum]"
fi

if [ -z ${EXAMPLE_NAME} ]
then
	request_example
fi

if [ -z ${JOB_CONFIG_PATH} ]
then
	JOB_CONFIG_ARG=""
else
	JOB_CONFIG_ARG="-f ${JOB_CONFIG_PATH}"
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
	*)
		request_example
		;;
esac

./run-csaf.sh ${JUPYTER} ${LOCAL} ${NATIVE} -d "${PWD}/${EXAMPLE_DIR}" -c ${CONF_NAME} ${JOB_CONFIG_ARG}

