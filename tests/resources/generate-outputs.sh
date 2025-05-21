#!/bin/bash
set -eux

echo "Generating data for testing..."

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd $ROOT_DIR

mtt train options.yaml -o model-32-bit.pt -r base_precision=32 # > /dev/null
mtt train options.yaml -o model-64-bit.pt -r base_precision=64 # > /dev/null
mtt train options-nanopet.yaml -o model-no-extensions.pt # > /dev/null

# upload results to private HF repo if token is set
if [ -n "${HUGGINGFACE_TOKEN_METATRAIN:-}" ]; then
    huggingface-cli upload \
        "metatensor/metatrain-test" \
        "model-32-bit.ckpt" \
        "model.ckpt" \
        --commit-message="Overwrite test model with new version" \
        --token=$HUGGINGFACE_TOKEN_METATRAIN
fi
