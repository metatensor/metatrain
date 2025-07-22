#!/bin/bash

if [[ $SLURM_PROCID == "0" ]]; then
  python3 -m pip install --upgrade pip
  python3 -m venv --system-site-packages venv
  source venv/bin/activate
  python3 -m pip install -e .
fi
