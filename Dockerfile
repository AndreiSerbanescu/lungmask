FROM nvidia/cuda:latest

WORKDIR /app

# Installing Python3 and Pip3
RUN apt-get update
RUN apt-get update && apt-get install -y python python-dev python3.7 python3.7-dev python3-pip virtualenv
RUN apt-get install -y libssl-dev libpq-dev git build-essential libfontconfig1 libfontconfig1-dev
RUN pip3 install setuptools pip --upgrade --force-reinstall

# Installing lungmask

# this is for avoiding Unicode decode error for the pip3 command
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
RUN apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    pip3 install git+https://github.com/bkainz/lungmask

RUN pip3 install SimpleITK

# Loading all three Unet models - R231, LTRCLobes and R231CovidWeb
COPY load_models.py /app/
RUN apt-get install vim -y #debugging

RUN python3 /app/load_models.py

COPY files/interface/ /app/

# COPY listen.py /app/listen.py
# RUN mkdir /app/common
# COPY common/* /app/common/