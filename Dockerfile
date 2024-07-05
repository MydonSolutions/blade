FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update --fix-missing

#
# This is copy-pasta from the README.md file.
# Update this as the README.md file changes.
#

RUN apt install -y git build-essential pkg-config cmake liberfa-dev libhdf5-dev libboost-all-dev libbenchmark-dev libgtest-dev python3-dev python3-pip
RUN python3 -m pip install meson ninja numpy astropy pandas

###

COPY . /blade
WORKDIR /blade

RUN git submodule update --init --recursive && \
meson setup /blade_build -Dprefix=/usr

WORKDIR /blade_build/
#RUN ninja install
