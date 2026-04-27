#!/usr/bin/env bash
# Move Exp01 results + artifacts (Panel A/B and shared build products, very large in size)
# to the data disk directory, with the original paths symlinked back for compatibility
# with existing scripts and document paths.
#
# Default target: /data/zd_data/KVI/exp01_main_qa/<timestamp>/{results,artifacts}
# If /data/zd_data is not writable, run first (requires admin, once):
#   sudo mkdir -p /data/zd_data && sudo chown -R "$(id -un):$(id -gn)" /data/zd_data
#
# Usage:
#   bash experiments/exp01_main_qa/code/relocate_exp01_panel_data_to_data_disk.sh
# Or:
#   DEST_ROOT=/data/zd_data/KVI bash experiments/exp01_main_qa/code/relocate_exp01_panel_data_to_data_disk.sh
#
# If directory permissions are insufficient, interactive sudo (will prompt for password):
#   USE_SUDO_MKDIR=1 bash experiments/exp01_main_qa/code/relocate_exp01_panel_data_to_data_disk.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP01_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEST_ROOT="${DEST_ROOT:-/data/zd_data/KVI}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${DEST_ROOT}/exp01_main_qa_${STAMP}"

RESULTS="${EXP01_ROOT}/results"
ARTIFACTS="${EXP01_ROOT}/artifacts"

log() { echo "[relocate] $*"; }

if [[ ! -d "${RESULTS}" ]]; then
  log "ERROR: missing ${RESULTS}"
  exit 2
fi
if [[ ! -d "${ARTIFACTS}" ]]; then
  log "ERROR: missing ${ARTIFACTS}"
  exit 2
fi

if [[ -L "${RESULTS}" ]]; then
  log "WARN: ${RESULTS} is already a symlink -> $(readlink -f "${RESULTS}" 2>/dev/null || readlink "${RESULTS}")"
  log "Nothing to do for results (already relocated?)."
  exit 0
fi

# Ensure destination root is writable (create archive dir)
if ! mkdir -p "${ARCHIVE}" 2>/dev/null; then
  if [[ "${USE_SUDO_MKDIR:-0}" == "1" ]]; then
    log "USE_SUDO_MKDIR=1: creating ${DEST_ROOT} with sudo ..."
    sudo mkdir -p "${DEST_ROOT}"
    sudo chown -R "$(id -un):$(id -gn)" "${DEST_ROOT}"
    mkdir -p "${ARCHIVE}"
  else
    log "ERROR: cannot create ${ARCHIVE}"
    log "Fix permissions, e.g.:"
    log "  sudo mkdir -p ${DEST_ROOT} && sudo chown -R $(id -un):$(id -gn) ${DEST_ROOT}"
    log "Or run: USE_SUDO_MKDIR=1 bash $0"
    exit 3
  fi
fi
rmdir "${ARCHIVE}" 2>/dev/null || true
mkdir -p "${ARCHIVE}"

log "Repo: ${REPO_ROOT}"
log "Exp01: ${EXP01_ROOT}"
log "Archive: ${ARCHIVE}"

# --- results ---
log "Moving results/ (~$(du -sh "${RESULTS}" | cut -f1)) ..."
mv "${RESULTS}" "${ARCHIVE}/results"
ln -s "${ARCHIVE}/results" "${RESULTS}"
log "Symlink: ${RESULTS} -> ${ARCHIVE}/results"

# --- artifacts ---
log "Moving artifacts/ (~$(du -sh "${ARTIFACTS}" | cut -f1)) ..."
mv "${ARTIFACTS}" "${ARCHIVE}/artifacts"
ln -s "${ARCHIVE}/artifacts" "${ARTIFACTS}"
log "Symlink: ${ARTIFACTS} -> ${ARCHIVE}/artifacts"

df -h "${DEST_ROOT}" "${EXP01_ROOT}" 2>/dev/null || true
log "Done. Panel A/B and shared artifacts moved to ${ARCHIVE}; original paths are symlinks."
