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
    fire \
    matplotlib \
    numpy \
    pexpect \
    pydot \
    pyqtgraph \
    pyros-genpy \
    scipy \
    toml \
    tqdm \
    zmq

RUN adduser --quiet --disabled-password qtuser

COPY src /app

WORKDIR /app

ENV PYTHONPATH /app
