#!/bin/bash
set -e

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd $ROOT_DIR

metatensor-models train options.yaml -o model-32-bit.pt -y base_precision=32
metatensor-models train options.yaml -o model-64-bit.pt -y base_precision=64
