#!/bin/bash
set -e

echo "Generate data for testing..."

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd $ROOT_DIR

metatensor-models --debug train options.yaml -o model-32-bit.pt -r base_precision=32 #> /dev/null
metatensor-models train options.yaml -o model-64-bit.pt -r base_precision=64 > /dev/null
