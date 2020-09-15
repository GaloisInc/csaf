FROM ubuntu:18.04

RUN apt-get clean && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
    git \
    python3 \
    python3-pip \
    graphviz \
    python3-pyqt5 \
    swig

RUN pip3 install -U pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN adduser --quiet --disabled-password qtuser

COPY src /app

WORKDIR /app

ENV PYTHONPATH /app
