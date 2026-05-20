#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zd/dev/KVI}"
PY="${ROOT}/KVI/bin/python3"
EXP2="${ROOT}/experiments/exp02_hallucination"
ART="${EXP2}/artifacts"
RES="${EXP2}/results"
MODEL="${MODEL:-${ROOT}/models/Qwen2.5-7B-Instruct}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

LOG="${RES}/exp02_p2_schema_build_and_run_${TS}.log"
PID_FILE="${RES}/exp02_p2_schema_build_and_run_${TS}.pid"

mkdir -p "${RES}"
cd "${ROOT}"
echo $$ > "${PID_FILE}"
echo "[$(date -Iseconds)] start P2 schema build+run ts=${TS}" | tee -a "${LOG}"

build_one() {
  local ds="$1"
  local ds_dir="${ART}/${ds}"
  local in_jsonl="${ds_dir}/sentences.tagged.jsonl"
  local out_schema="${ds_dir}/blocks.schema.jsonl"
  local out_kv="${ds_dir}/kvbank_schema"

  echo "[$(date -Iseconds)] [${ds}] build blocks.schema.jsonl" | tee -a "${LOG}"
  "${PY}" "${ROOT}/scripts/build_schema_blocks_from_evidence_jsonl.py" \
    --blocks_jsonl_evidence "${in_jsonl}" \
    --out_jsonl "${out_schema}" \
    >> "${LOG}" 2>&1

  echo "[$(date -Iseconds)] [${ds}] build kvbank_schema" | tee -a "${LOG}"
  "${PY}" "${ROOT}/scripts/build_kvbank_from_blocks_jsonl.py" \
    --blocks_jsonl "${out_schema}" \
    --out_dir "${out_kv}" \
    --base_llm "${MODEL}" \
    --domain_encoder_model "sentence-transformers/all-MiniLM-L6-v2" \
    --layers "0,1,2,3" \
    --block_tokens 128 \
    --shard_size 1024 \
    --device cpu \
    --dtype float32 \
    >> "${LOG}" 2>&1
}

build_one "truthfulqa"
build_one "fever"

echo "[$(date -Iseconds)] launch vNext protocol detached (P2-wired)" | tee -a "${LOG}"
nohup env TS="${TS}" ROOT="${ROOT}" MODEL="${MODEL}" RESIDENT_URL="${RESIDENT_URL:-http://127.0.0.1:18888}" \
  bash "${EXP2}/code/run_vnext_protocol_detached.sh" </dev/null \
  >> "${RES}/exp02_vnext_protocol_outer.log" 2>&1 &
echo "[$(date -Iseconds)] started vNext pid=$!" | tee -a "${LOG}"

echo "[$(date -Iseconds)] done P2 schema build+run ts=${TS}" | tee -a "${LOG}"
