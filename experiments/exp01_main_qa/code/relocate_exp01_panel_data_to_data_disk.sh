#!/usr/bin/env bash
# Move Exp01 results + artifacts (Panel A/B 及共用构建产物，体积极大) 到大盘目录，
# 原路径用符号链接指回，便于脚本与文档路径不变。
#
# 默认目标: /data/zd_data/KVI/exp01_main_qa/<timestamp>/{results,artifacts}
# 若 /data/zd_data 不可写，先执行（需管理员一次）:
#   sudo mkdir -p /data/zd_data && sudo chown -R "$(id -un):$(id -gn)" /data/zd_data
#
# 用法:
#   bash experiments/exp01_main_qa/code/relocate_exp01_panel_data_to_data_disk.sh
# 或:
#   DEST_ROOT=/data/zd_data/KVI bash experiments/exp01_main_qa/code/relocate_exp01_panel_data_to_data_disk.sh
#
# 若目录权限不足，可交互式 sudo（会提示密码）:
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
log "Done. Panel A/B 与共用 artifacts 已迁至 ${ARCHIVE}；原路径为符号链接。"
