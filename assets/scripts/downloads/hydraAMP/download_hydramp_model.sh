#!/usr/bin/env bash
#
# Installs one named HydrAMP model variant (encoder/decoder weights) into
# src/pep_compass/autoencoder/strategies/hydramp/models/<model>/.
#
# The 'article_25' variant already ships committed in this repository and does
# not need this script. Use it for additional variants once their source is
# known. No default source repository is hardcoded here; provide one via
# HYDRAMP_REPOSITORY / HYDRAMP_SOURCE_PATH before running.
#
# Usage:
#   HYDRAMP_REPOSITORY=<git-url> HYDRAMP_SOURCE_PATH=<path-in-repo> \
#     ./download_hydramp_model.sh <model-name>

set -euo pipefail

MODEL_NAME="${1:-article_25}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
TARGET_DIR="${HYDRAMP_MODELS_DIR:-${REPOSITORY_ROOT}/src/pep_compass/autoencoder/strategies/hydramp/models/${MODEL_NAME}}"
TEMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

if [[ -n "$(find "${TARGET_DIR}" -maxdepth 1 -name '*.pickle' -print -quit 2>/dev/null)" ]]; then
    echo "HydrAMP model '${MODEL_NAME}' weights already present in: ${TARGET_DIR}" >&2
    echo "Skipping download. Remove the files explicitly to force a fresh copy." >&2
    exit 0
fi

if [[ -z "${HYDRAMP_REPOSITORY:-}" ]]; then
    echo "HYDRAMP_REPOSITORY is not set. Provide the source repository for" >&2
    echo "HydrAMP model '${MODEL_NAME}' weights before running this script." >&2
    exit 1
fi
if [[ -z "${HYDRAMP_SOURCE_PATH:-}" ]]; then
    echo "HYDRAMP_SOURCE_PATH is not set. Provide the path to the model" >&2
    echo "directory inside ${HYDRAMP_REPOSITORY} before running this script." >&2
    exit 1
fi

git clone --depth 1 --filter=blob:none --sparse \
    "${HYDRAMP_REPOSITORY}" "${TEMP_DIR}/hydramp"
git -C "${TEMP_DIR}/hydramp" sparse-checkout set \
    "${HYDRAMP_SOURCE_PATH}"

SOURCE_DIR="${TEMP_DIR}/hydramp/${HYDRAMP_SOURCE_PATH}"
if [[ ! -d "${SOURCE_DIR}" ]] || [[ -z "$(find "${SOURCE_DIR}" -type f -print -quit)" ]]; then
    echo "HydrAMP model '${MODEL_NAME}' weights were not found in the downloaded repository." >&2
    exit 1
fi

mkdir -p -- "${TARGET_DIR}"
cp -R -- "${SOURCE_DIR}/." "${TARGET_DIR}/"

echo "HydrAMP model '${MODEL_NAME}' weights installed in: ${TARGET_DIR}"
