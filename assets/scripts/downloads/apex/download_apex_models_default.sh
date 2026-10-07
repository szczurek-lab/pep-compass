#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
APEXGO_REPOSITORY="${APEXGO_REPOSITORY:-https://github.com/Yimeng-Zeng/APEXGo.git}"
TARGET_DIR="${APEX_MODELS_DIR:-${REPOSITORY_ROOT}/src/pep_compass/optimization/components/oracles/strategies/apex/models/default}"
TEMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

if [[ -n "$(find "${TARGET_DIR}" -maxdepth 1 -name 'APEX_*' -print -quit 2>/dev/null)" ]]; then
    echo "APEX 'default' model weights already present in: ${TARGET_DIR}" >&2
    echo "Skipping download. Remove the files explicitly to force a fresh copy." >&2
    exit 0
fi

# Initialize sparse checkout separately to support Git 2.25.1.
git clone --depth 1 --filter=blob:none --no-checkout \
    "${APEXGO_REPOSITORY}" "${TEMP_DIR}/APEXGo"

git -C "${TEMP_DIR}/APEXGo" sparse-checkout init --cone
git -C "${TEMP_DIR}/APEXGo" sparse-checkout set \
    optimization/apex_oracle/APEX_pathogen_models
git -C "${TEMP_DIR}/APEXGo" checkout

SOURCE_DIR="${TEMP_DIR}/APEXGo/optimization/apex_oracle/APEX_pathogen_models"
if [[ ! -d "${SOURCE_DIR}" ]] || [[ -z "$(find "${SOURCE_DIR}" -type f -print -quit)" ]]; then
    echo "APEX 'default' model weights were not found in the downloaded repository." >&2
    exit 1
fi

mkdir -p -- "${TARGET_DIR}"
cp -R -- "${SOURCE_DIR}/." "${TARGET_DIR}/"

echo "APEX 'default' model weights installed in: ${TARGET_DIR}"