FROM docker://nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# FFTW library required by spinfast: https://github.com/moble/spinsfast?tab=readme-ov-file#pip
RUN apt update && \
	apt install -y libfftw3-dev python3 python3-pip python3-venv git && \
	apt clean

ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128

COPY . /mtt-repo/

RUN python3 -m venv --system-site-packages /mtt-venv && \
    . /mtt-venv/bin/activate && \
    python -m pip install /mtt-repo[soap-bpnn]

ENV CUDA_HOME="/usr/local/cuda"
