FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

COPY dependencies.sh /tmp/deps.sh
RUN /tmp/deps.sh

RUN pip3 install -U pip
RUN pip3 install --upgrade pip
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
