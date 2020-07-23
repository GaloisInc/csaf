FROM ubuntu:18.04

RUN apt-get clean && \
    apt-get update && \
    apt-get -y upgrade

RUN apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-pyqt5

RUN pip3 install -U pip
RUN pip3 install \
    matplotlib \
    numpy \
    pexpect \
    pyqtgraph \
    scipy \
    toml \
    zmq

RUN adduser --quiet --disabled-password qtuser

WORKDIR /app

ENV PYTHONPATH /app
