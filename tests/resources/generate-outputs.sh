#!/bin/bash
set -eux

echo "Generating data for testing..."

# Define all model parameters in an associative array (like a dictionary)
# Key: The output filename
# Value: A semicolon-separated string of "config_file;extra_arg1;extra_arg2;..."
declare -A models=(
    ["model-32-bit.pt"]="options.yaml;-r;base_precision=32"
    ["model-64-bit.pt"]="options.yaml;-r;base_precision=64"
    ["model-pet.pt"]="options-pet.yaml"
)

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd "$ROOT_DIR"

HASH_FILE=".data_version.txt"
# Things
WATCH_PATHS="src/"
FORCE_REGENERATE=false
if [[ "${FORCE_REGENERATE:-0}" == "1" ]]; then
  echo "FORCE_REGENERATE=1 detected. Forcing regeneration of all models."
  FORCE_REGENERATE=true
elif [ -n "$(git status --porcelain -- $WATCH_PATHS)" ]; then
  echo "Uncommitted git changes detected in critical files. Regenerating."
  FORCE_REGENERATE=true
else
  if [ -f "$HASH_FILE" ]; then
    SAVED_HASH=$(cat "$HASH_FILE")
    CURRENT_HASH=$(git rev-parse HEAD)
    if [ "$SAVED_HASH" != "$CURRENT_HASH" ]; then
      echo "Git commit has changed. Forcing regeneration of all models."
      FORCE_REGENERATE=true
    fi
  else
    echo "Hash file not found. Forcing regeneration of all models."
    FORCE_REGENERATE=true
  fi
fi

# Regenerate if --force is used OR if the file doesn't exist
for model_file in "${!models[@]}"; do
    # Regenerate if --force is used OR if the file doesn't exist
    if [ "$FORCE_REGENERATE" = true ] || [ ! -f "$model_file" ]; then
        echo "Generating '$model_file'..."

        # Read the parameter string for the current model
        params_str=${models["$model_file"]}

        # Safely split the string into a temporary array using ';' as the delimiter
        IFS=';' read -r -a params_array <<< "$params_str"

        # The first element is the config file
        config_file=${params_array[0]}

        # The rest of the elements are extra arguments for the command
        extra_args=("${params_array[@]:1}")

        # Execute the command, safely passing the arguments
        mtt train "$config_file" -o "$model_file" "${extra_args[@]}"
    fi
done

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
