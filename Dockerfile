FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

COPY dependencies.sh /tmp/deps.sh
RUN /tmp/deps.sh

RUN pip3 install -U pip
RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN adduser --quiet --disabled-password qtuser

WORKDIR /

ENV PYTHONPATH /app
