#!/bin/bash
set -eux

echo "Generating data for testing..."

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd "$ROOT_DIR"

HASH_FILE=".data_version.txt"
WATCH_PATHS="src/"
FORCE_REGENERATE=true

if [[ "${USE_CACHE:-0}" == "1" ]]; then
    echo "USE_CACHE=1 detected. Attempting to use cached data."
    CACHE_IS_VALID=true
    if [ -n "$(git status --porcelain -- $WATCH_PATHS)" ]; then
        echo "Cache is invalid due to uncommitted changes. Must regenerate."
        CACHE_IS_VALID=false
    elif [ ! -f "$HASH_FILE" ]; then
        echo "Cache is invalid: version file not found. Must regenerate."
        CACHE_IS_VALID=false
    else
        SAVED_HASH=$(cat "$HASH_FILE")
        CURRENT_HASH=$(git rev-parse HEAD)
        if [ "$SAVED_HASH" != "$CURRENT_HASH" ]; then
            echo "Cache is invalid: code version has changed. Must regenerate."
            CACHE_IS_VALID=false
        fi
    fi

    # If all checks passed, we can rely on the cache.
    if [ "$CACHE_IS_VALID" = true ]; then
        echo "Cache is valid. Will skip regeneration for existing files."
        FORCE_REGENERATE=false
    fi
fi

# Regenerate if regeneration is forced (default) OR if a file is missing.
if [ "$FORCE_REGENERATE" = true ] || [ ! -f "model-32-bit.pt" ]; then
    mtt train options.yaml -o model-32-bit.pt -r base_precision=32
fi

if [ "$FORCE_REGENERATE" = true ] || [ ! -f "model-64-bit.pt" ]; then
    mtt train options.yaml -o model-64-bit.pt -r base_precision=64
fi

if [ "$FORCE_REGENERATE" = true ] || [ ! -f "model-pet.pt" ]; then
    mtt train options-pet.yaml -o model-pet.pt
fi

if [ "$FORCE_REGENERATE" = true ]; then
  echo "Saving current git commit hash to version the data."
  git rev-parse HEAD > "$HASH_FILE"
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
