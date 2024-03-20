#/bin/bash

cd `dirname "$(realpath $0)"`

metatensor-models train options.yaml -o model-32-bit.pt -y base_precision=32
metatensor-models train options.yaml -o model-64-bit.pt -y base_precision=64
