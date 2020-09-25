#!/bin/bash
JUPYTER=0
source .common.sh

validate_dir

print_help() {
    printf "\033[1mCSAF\033[0m\n"
    printf "    Control System Analysis Framework (CSAF) is a middleware framework that
    makes creating and evaluating control systems as simple as possible. Control
    loop topologies and component implementations are specified independently of
    the middleware.\n"
    printf "\n\033[1mUSAGE\033[0m\n"
    printf "   -e      the name of the example { f16-shield, f16-simple, f16-llc, inv-pendulum }\n"
    printf "   -c      the name of the model config file (must be in the same directory as your system)\n"
    printf "   -d      fully qualified path to the directory defining the model system\n"
    printf "   -f      name of the job config file (must be in the same directory as your system)\n"
    printf "   -j      launch a jupyter notebook\n"
    printf "   -l      build the image locally\n"
    printf "   -n      run CSAF natively\n"
    printf "   -t      the tag of the image { stable, edge, latest }\n"
    printf "   -h      prints the help menu\n"
    printf "   -x      clear the output for a particular example/config"
    printf "\n\n"
    printf "\033[1mEXAMPLES\033[0m\n"
    printf "Run f16-simple example:\n"
    printf "    ./run-csaf.sh -e f16-simple\n"
    printf "Run f16-simple example natively (not in a docker container):\n"
    printf "    ./run-csaf.sh -e f16-simple -n\n"
    printf "Start a jupyter notebook with f16 example:\n"
    printf "    ./run-csaf.sh -e f16-simple -j\n"
    printf "    ./run-csaf.sh -e f16-shield -j\n"
    printf "    ./run-csaf.sh -d \${PWD}/examples/f16 -j\n"
    printf "Start jupyter notebook with your own example:\n"
    printf "    ./run-csaf.sh -j -d \${PWD}/examples/inverted-pendulum\n"
    printf "Run f16-shield with your own job config:\n"
    printf "    ./run-csaf.sh -e f16-shield -f f16_job_conf.toml\n"
    printf "Clear generated outputs for f16 example:\n"
    printf "    ./run-csaf.sh -e f16-simple -x\n"
}

while getopts ":c:d:e:f:t:jlhnx:" opt; do
    case ${opt} in
        c )
            CONFIG_NAME=`basename $OPTARG`
            ;;
        d )
            CSAF_LOC=$OPTARG
            ;;
        e )
            EXAMPLE_NAME=$OPTARG
            ;;
        f )
            JOB_CONFIG_PATH=`basename $OPTARG`
            ;;
        j )
            JUPYTER=1
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
        x )
            CLEAR_OUTPUT=1
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

if [[ -n ${EXAMPLE_NAME} && -n ${CONFIG_NAME} ]] ;
then
    show_error_and_exit "A CSAF config file and an example name supplied at the same time. Please choose only one."
fi

if [ ! -z ${EXAMPLE_NAME} ] ;
then
    case $EXAMPLE_NAME in
        "f16-simple")
            CONFIG_NAME="f16_simple_config.toml"
            CSAF_LOC=${PWD}/"examples/f16"
            ;;
        "f16-shield")
            CONFIG_NAME="f16_shield_config.toml"
            CSAF_LOC=${PWD}/"examples/f16"
            ;;
        "inv-pendulum")
            CONFIG_NAME="inv_pendulum_config.toml"
            CSAF_LOC=${PWD}/"examples/inverted-pendulum"
            ;;
	"f16-llc")
            CONFIG_NAME="f16_llc_analyze_config.toml"
            CSAF_LOC=${PWD}/"examples/f16"
            ;;
        *)
            show_error_and_exit "Unknown example: ${EXAMPLE_NAME} Please use one of [f16-shield, f16-simple, inv-pendulum]"
            ;;
    esac
fi


if [[ ${JUPYTER} -eq 0 && -z ${CONFIG_NAME} ]] ;
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

if [[ ${JUPYTER} -eq 1 && ${NATIVE} -eq 1  ]] ; then
    show_error_and_exit "the \'native\' and \'jupyter\' options cannot be combined"
fi

if [[ ${LOCAL} -eq 1 ]] ; then
    show_info "Building docker image locally."
    build_img
fi

if [[ ${CLEAR_OUTPUT} -eq 1 ]] ; then
    echo "Remove ${CSAF_LOC}/output and ${CSAF_LOC}/codec? [Y/n]"
    read ACK
    if [ ${ACK} = "Y" ]; then
        rm -rf ${CSAF_LOC}/output
        rm -rf ${CSAF_LOC}/codec
        echo "Done"
    else
        echo "Aborting."
    fi
    exit 0
fi

if [[ ${NATIVE} -eq 1 ]] ; then
    if [[ ${JUPYTER} -eq 1 ]] ; then
        jupyter notebook --no-browser --notebook-dir=${PWD}/docs/notebooks
    else
        python3 "src/run_system.py" ${CSAF_LOC} ${CONFIG_NAME} ${JOB_CONFIG_PATH}
    fi
else
    if [[ ${JUPYTER} -eq 1 ]] ; then
        docker run --init --network host -it -v ${CSAF_LOC}:/csaf-system \
            -v ${PWD}/src:/app -v ${PWD}/docs/notebooks:/notebooks \
            ${IMAGE_NAME}:${IMAGE_TAG} "jupyter" "notebook" "--port=8888" \
            "--no-browser" "--ip=0.0.0.0" "--allow-root" "--notebook-dir=/notebooks"
    else
        docker run --init -it -v ${PWD}/src:/app -v ${CSAF_LOC}:/csaf-system --network host \
            ${IMAGE_NAME}:${IMAGE_TAG} python3 "/app/run_system.py" "/csaf-system" ${CONFIG_NAME} ${JOB_CONFIG_PATH}
    fi
fi

