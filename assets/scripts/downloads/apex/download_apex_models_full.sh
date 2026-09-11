#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
APEX_REPOSITORY="${APEX_REPOSITORY:-https://gitlab.com/machine-biology-group-public/apex.git}"
TARGET_DIR="${APEX_MODELS_DIR:-${REPOSITORY_ROOT}/src/pep_compass/optimization/components/oracles/strategies/apex/models/full}"
TEMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

if [[ -n "$(find "${TARGET_DIR}" -maxdepth 1 -name 'trained_*' -print -quit 2>/dev/null)" ]]; then
    echo "APEX 'full' model weights already present in: ${TARGET_DIR}" >&2
    echo "Skipping download. Remove the files explicitly to force a fresh copy." >&2
    exit 0
fi

git clone --depth 1 --filter=blob:none --sparse \
    "${APEX_REPOSITORY}" "${TEMP_DIR}/apex"
git -C "${TEMP_DIR}/apex" sparse-checkout set \
    trained_models

SOURCE_DIR="${TEMP_DIR}/apex/trained_models"
if [[ ! -d "${SOURCE_DIR}" ]] || [[ -z "$(find "${SOURCE_DIR}" -type f -print -quit)" ]]; then
    echo "APEX 'full' model weights were not found in the downloaded repository." >&2
    exit 1
fi

mkdir -p -- "${TARGET_DIR}"
cp -R -- "${SOURCE_DIR}/." "${TARGET_DIR}/"

echo "APEX 'full' model weights installed in: ${TARGET_DIR}"
