FROM ubuntu:18.04

# Add user
RUN adduser --quiet --disabled-password qtuser

# Install Python 3, PyQt5
RUN apt-get update \
    && apt-get install -y \
      python3 \
      python3-pyqt5

# install required programs
RUN apt-get -y install git

# setup the python installation
RUN apt-get -y install python3-pip
RUN pip3 install -U pip
RUN pip3 install zmq 
RUN pip3 install matplotlib 
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install pyqtgraph
RUN pip3 install toml
RUN pip3 install pexpect

WORKDIR /app

ENV PYTHONPATH /app
