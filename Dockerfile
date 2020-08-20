FROM ubuntu:18.04

RUN apt-get clean && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
    git \
    python3 \
    python3-pip \
    graphviz \
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
    zmq \
    pymap3d

RUN adduser --quiet --disabled-password qtuser

COPY src /app

WORKDIR /app

ENV PYTHONPATH /app
