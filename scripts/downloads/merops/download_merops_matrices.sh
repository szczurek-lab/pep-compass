#!/usr/bin/env bash
#
# Installs the MEROPS peptidase specificity matrices into data/merops/.
#
# These are small, curated matrices (index.json + per-dataset
# matrices_logprob.npy / matrices_counts.npy / codes.json) consumed by
# pep_compass.optimization.components.helpers.proteolysis.load_protease_panel.
# They are kept out of git (data/* is gitignored) and fetched with this script,
# mirroring the APEX / HydrAMP model download scripts.
#
# No source repository is hardcoded here (the matrices are derived from the
# MEROPS 2023 Substrate_search release and are not published in a fixed public
# location). Provide the source via MEROPS_REPOSITORY before running, following
# the same convention as download_hydramp_model.sh.
#
# Usage:
#   MEROPS_REPOSITORY=<git-url> [MEROPS_SOURCE_PATH=<path-in-repo>] \
#     ./download_merops_matrices.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
# Path to the merops data directory inside the source repository.
SOURCE_PATH="${MEROPS_SOURCE_PATH:-data/merops}"
TARGET_DIR="${MEROPS_DATA_DIR:-${REPOSITORY_ROOT}/data/merops}"
TEMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

if [[ -f "${TARGET_DIR}/index.json" ]]; then
    echo "MEROPS matrices already present in: ${TARGET_DIR}" >&2
    echo "Skipping download. Remove the files explicitly to force a fresh copy." >&2
    exit 0
fi

if [[ -z "${MEROPS_REPOSITORY:-}" ]]; then
    echo "MEROPS_REPOSITORY is not set. Provide the source repository holding" >&2
    echo "the MEROPS specificity matrices (a '${SOURCE_PATH}' directory with" >&2
    echo "index.json and per-dataset matrices) before running this script." >&2
    exit 1
fi

git clone --depth 1 --filter=blob:none --sparse \
    "${MEROPS_REPOSITORY}" "${TEMP_DIR}/merops"
git -C "${TEMP_DIR}/merops" sparse-checkout set \
    "${SOURCE_PATH}"

SOURCE_DIR="${TEMP_DIR}/merops/${SOURCE_PATH}"
if [[ ! -f "${SOURCE_DIR}/index.json" ]]; then
    echo "MEROPS matrices (index.json) were not found in the downloaded" >&2
    echo "repository under '${SOURCE_PATH}'." >&2
    exit 1
fi

mkdir -p -- "${TARGET_DIR}"
cp -R -- "${SOURCE_DIR}/." "${TARGET_DIR}/"

echo "MEROPS matrices installed in: ${TARGET_DIR}"
