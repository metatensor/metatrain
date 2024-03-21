#!/bin/bash
set -e

cd `dirname "$(realpath $0)"`

pwd

metatensor-models train options.yaml -o model-32-bit.pt -y base_precision=32
metatensor-models train options.yaml -o model-64-bit.pt -y base_precision=64
