FROM ubuntu:21.04

ARG DEBIAN_FRONTEND=noninteractive

COPY dependencies.sh /tmp/deps.sh
RUN /tmp/deps.sh

#COPY requirements.txt /tmp/requirements.txt
COPY new_csaf/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN adduser --quiet --disabled-password qtuser

WORKDIR /

ENV PYTHONPATH /app
