# CSAF Docker Image

# set base image (host OS)
FROM python:3.9

# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# install csaf + examples
COPY . csaf/
RUN pip install csaf/
