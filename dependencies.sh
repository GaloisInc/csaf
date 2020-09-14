#!/bin/bash
apt-get clean && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
    git \
    python3 \
    python3-pip \
    graphviz \
    python3-pyqt5 \
    pandoc

pip3 install -U pip
pip3 install --upgrade pip

# Latex for notebooks
apt-get install -y texlive-xetex texlive-fonts-recommended texlive-generic-recommended
apt-get -y install pandoc