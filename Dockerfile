FROM ubuntu:18.04

RUN apt-get clean && \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-pyqt5 \
    graphviz

RUN pip3 install -U pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install jupyter

RUN adduser --quiet --disabled-password qtuser

COPY docs/notebooks /notebooks
COPY src /app

WORKDIR /

ENV PYTHONPATH /app

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
