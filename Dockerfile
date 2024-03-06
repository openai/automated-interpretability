FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip


COPY . /app
WORKDIR /app

# make sure the working dir made already
# docker build -t yourdockerimage .
RUN pip install jupyter && \
    cd neuron-explainer && \
    pip install -e .

# to run normally
# docker run -it --name yourdockercontainer yourdockerimage
# to run out of jupyter
# docker run -it -p 8888:8888 --name yourdockercontainer yourdockerimage
# select kernel and input url from this