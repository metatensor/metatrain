#!/bin/bash
set -eux

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODE=$1
TRAIN_ID=$2

if [ -z "$MODE" ]; then
    echo "Error: First argument of the script is the mode"
    echo " There is no first argument, please set it to "
    echo " '32-bit', '64-bit' or 'pet'"
    exit 1
fi

# If there is a TRAIN_ID, implement a lock file
if [ -n "$TRAIN_ID" ]; then
    echo "Creating lockfile"
    LOCKFILE="$ROOT_DIR/$MODE-$TRAIN_ID.trainlock"
    if [ -f $LOCKFILE ]; then
        # Wait until the lock file is removed
        while [ -f $LOCKFILE ]; do
            sleep 5
        done
        exit 0
    else
        touch $LOCKFILE
    fi
fi

echo "Clearing previous generated files..."
# Clean previous generated files
rm $ROOT_DIR/model-$MODE-*.pt $ROOT_DIR/model-$MODE-*.ckpt || true
rm $ROOT_DIR/$MODE-*.trainlock || true

echo "Generating data for testing..."

cd "$ROOT_DIR"
# The generated files are uniquely identified by the TRAIN_ID passed as second argument,
# in this way a test run can know if the files have already been generated.
if [ "$MODE" == "32-bit" ]; then
    mtt train options.yaml -o model-32-bit-$TRAIN_ID.pt -r base_precision=32
elif [ "$MODE" == "64-bit" ]; then
    mtt train options.yaml -o model-64-bit-$TRAIN_ID.pt -r base_precision=64
elif [ "$MODE" == "pet" ]; then
    mtt train options-pet.yaml -o model-pet-$TRAIN_ID.pt
else
    echo "Error: Unknown training mode (first argument): '$MODE'"
    echo " Please set it to '32-bit', '64-bit' or 'pet'"
    exit 1
fi

rm $LOCKFILE || true

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
        "model-32-bit-$TRAIN_ID.ckpt" \
        "model.ckpt" \
        --commit-message="Overwrite test model with new version" \
        --token="$HUGGINGFACE_TOKEN_METATRAIN"
else
    echo "HUGGINGFACE_TOKEN_METATRAIN is not set, skipping upload."
fi
