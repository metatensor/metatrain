FROM nvcr.io/nvidia/pytorch:25.02-py3

# FFTW library required by spinfast: https://github.com/moble/spinsfast?tab=readme-ov-file#pip
RUN apt update && \
    apt install -y libfftw3-dev && \
    apt clean

RUN python3 -m pip install --upgrade pip setuptools packaging

COPY . /mtt-repo/

RUN python3 -m venv --system-site-packages /mtt-venv && \
    source /mtt-venv/bin/activate && \
    which python && \
    python -m pip install --no-binary=metatensor-torch --no-build-isolation metatensor-torch && \
    python -m pip install /mtt-repo[soap-bpnn]

ENV CUDA_HOME="/usr/local/cuda"
