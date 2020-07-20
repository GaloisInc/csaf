#!/bin/bash
IMAGE_NAME=elew/csaf-devel
IMAGE_TAG=csaf

unameOut="$(uname -s)"
case "${unameOut}" in
	Linux*)	SUDO=sudo;;
	Darwin*) SUDO=;;
	*) echo "Unknown Machine"; exit 1;;
esac


echo "Start building $IMAGE_NAME:$IMAGE_TAG Image"
$SUDO docker build -t $IMAGE_NAME:$IMAGE_TAG .
if [[ $? -ne 0 ]]
then
	echo "Error: Create Image Locally"
	exit 1
fi

$SUDO docker push elew/beatpdx:tagname
echo "Publish the image"
$SUDO docker push $IMAGE_NAME:$IMAGE_TAG
if [[ $? -ne 0 ]]
then
	echo "Error: Publish the image"
	exit 1
fi

echo "Docker $IMAGE_NAME:$IMAGE_TAG installed and published successfully"

