#!/bin/bash
set -eux

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODE=$1

if [ -z "$MODE" ]; then
    echo "Error: First argument of the script is the mode"
    echo " There is no first argument, please set it to "
    echo " '32-bit', '64-bit' or 'pet'"
    exit 1
fi

echo "Generating data for testing..."

cd "$ROOT_DIR"

TRAIN_DIR="$ROOT_DIR/train_$MODE"
rm -r "$TRAIN_DIR" || true
mkdir -p "$TRAIN_DIR"
cp *yaml *xyz "$TRAIN_DIR" 
cd "$TRAIN_DIR"
if [ "$MODE" == "32-bit" ]; then
    mtt train ../options.yaml -o model-32-bit.pt -r base_precision=32
elif [ "$MODE" == "64-bit" ]; then
    mtt train ../options.yaml -o model-64-bit.pt -r base_precision=64
elif [ "$MODE" == "pet" ]; then
    mtt train ../options-pet.yaml -o model-pet.pt
else
    echo "Error: Unknown training mode (first argument): '$MODE'"
    echo " Please set it to '32-bit', '64-bit' or 'pet'"
    exit 1
fi

# If the mode is 32-bit, we will try to upload the model to Hugging Face,
# otherwise we are done here
if [ "$MODE" != "32-bit" ]; then
    exit 0
fi

set +x  # disable command echoing for sensitive private token check
TOKEN_PRESENT=false
if [[ -n "${HUGGINGFACE_TOKEN_METATRAIN:-}" ]]; then
    TOKEN_PRESENT=true
fi
set -x

if [ $TOKEN_PRESENT = true ]; then
    hf upload \
        "metatensor/metatrain-test" \
        "model-32-bit.ckpt" \
        "model.ckpt" \
        --commit-message="Overwrite test model with new version" \
        --token="$HUGGINGFACE_TOKEN_METATRAIN"
else
    echo "HUGGINGFACE_TOKEN_METATRAIN is not set, skipping upload."
fi
