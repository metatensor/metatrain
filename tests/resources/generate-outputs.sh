#!/bin/bash
set -eux

echo "Generating data for testing..."

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd "$ROOT_DIR"

FORCE_REGENERATE=false
if [[ "${FORCE_REGENERATE:-0}" == "1" ]]; then
  echo "FORCE_REGENERATE=1 detected. Regenerating all models."
  FORCE_REGENERATE=true
fi

# Regenerate if --force is used OR if the file doesn't exist
if [ "$FORCE_REGENERATE" = true ] || [ ! -f "model-32-bit.pt" ]; then
    mtt train options.yaml -o model-32-bit.pt -r base_precision=32
fi

if [ "$FORCE_REGENERATE" = true ] || [ ! -f "model-64-bit.pt" ]; then
    mtt train options.yaml -o model-64-bit.pt -r base_precision=64
fi

if [ "$FORCE_REGENERATE" = true ] || [ ! -f "model-pet.pt" ]; then
    mtt train options-pet.yaml -o model-pet.pt
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
