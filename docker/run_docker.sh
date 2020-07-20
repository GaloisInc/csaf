#!/bin/bash
IMAGE_NAME=elew/beatpdx
IMAGE_TAG=beatpd

# requirement -- correctly extracted BEAT PD data AND submission files
if [ -z "$BEAT_PD_DIR" ]
then
	echo "you must define variable BEAT_PD_DIR!"
	exit 1
fi

# enforce directory the script is run in
if [ `basename $PWD` != "beat-pd-pdx" ]
then
	echo "must run script from repo root beat-pd-pdx"
	exit 1
fi

# check for sudo command
unameOut="$(uname -s)"
case "${unameOut}" in
	Linux*)	SUDO=sudo;;
	Darwin*) SUDO=;;
	*) echo "Unknown Machine"; exit 1;;
esac

$SUDO docker run --network host -it -v $PWD:/beat-pd-pdx -v $BEAT_PD_DIR:/datasets/beat-pd $IMAGE_NAME:$IMAGE_TAG
