FROM nvcr.io/nvidia/pytorch:25.02-py3

# FFTW library required by spinfast: https://github.com/moble/spinsfast?tab=readme-ov-file#pip
RUN apt update && \
	apt install -y libfftw3-dev && \
	apt clean

RUN python3 -m pip install --upgrade pip setuptools

ENV CUDA_HOME="/usr/local/cuda"
