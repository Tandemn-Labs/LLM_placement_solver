#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run LLM Advisor multiple times and capture outputs in CSV.

Usage:
  ./run-llm-advisor-batch.sh -n 10 -p openai -l gpt-5.2 -o runs.csv -k "$OPENAI_API_KEY"

Options:
  -n, --runs        Number of runs (default: 5)
  -o, --out         Output CSV path (default: advisor_runs.csv)
  -p, --provider    LLM provider: anthropic|openai (default: openai)
  -l, --llm-model   Advisor LLM model name (default: gpt-5.2)
  -k, --api-key     API key (optional; falls back to env)
  -r, --run-script  Path to run-llm-advisor.sh (default: ./run-llm-advisor.sh)
  --log-dir         Directory to store raw logs (default: advisor_runs)
  --sleep           Seconds to sleep between runs (default: 0)
  -h, --help        Show help

Notes:
  - The target deployment model is still whatever run-llm-advisor.sh uses
    (currently hardcoded to llama-70b).
USAGE
}

RUNS=5
OUT_CSV="advisor_runs.csv"
LOG_DIR="advisor_runs"
PROVIDER="openai"
LLM_MODEL="gpt-5.2"
API_KEY=""
RUN_SCRIPT="./run-llm-advisor.sh"
SLEEP_SEC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--runs) RUNS="$2"; shift 2;;
    -o|--out) OUT_CSV="$2"; shift 2;;
    -p|--provider) PROVIDER="$2"; shift 2;;
    -l|--llm-model) LLM_MODEL="$2"; shift 2;;
    -k|--api-key) API_KEY="$2"; shift 2;;
    -r|--run-script) RUN_SCRIPT="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --sleep) SLEEP_SEC="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$API_KEY" ]]; then
  if [[ "$PROVIDER" == "openai" ]]; then
    API_KEY="${OPENAI_API_KEY:-}"
  else
    API_KEY="${ANTHROPIC_API_KEY:-}"
  fi
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Run script not found: $RUN_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

if [[ ! -f "$OUT_CSV" ]]; then
  printf '"run_id","timestamp","provider","llm_model","recommendation","confidence","warnings","critical_analysis","raw_log"\n' > "$OUT_CSV"
fi

csv_escape() {
  local s="$1"
  s="${s//$'\r'/}"
  s="${s//\"/\"\"}"
  s="${s//$'\n'/\\n}"
  printf '%s' "$s"
}

for ((i=1; i<=RUNS; i++)); do
  ts=$(date -Iseconds)
  log="${LOG_DIR}/run_${i}.txt"

  "$RUN_SCRIPT" "$API_KEY" "$PROVIDER" "$LLM_MODEL" > "$log" 2>&1

  rec_line=$(grep -m1 '^RECOMMENDATION:' "$log" || true)
  conf_line=$(grep -m1 '^Confidence:' "$log" || true)
  warn_line=$(grep -m1 '^Warnings:' "$log" || true)

  rec="${rec_line#RECOMMENDATION: }"
  conf="${conf_line#Confidence: }"
  warn="${warn_line#Warnings: }"

  crit=$(awk 'BEGIN{found=0} { if (found) print; if (tolower($0) ~ /critical analysis/) {found=1; print;} }' "$log" || true)

  printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
    "$i" "$ts" "$PROVIDER" "$LLM_MODEL" \
    "$(csv_escape "$rec")" "$(csv_escape "$conf")" "$(csv_escape "$warn")" \
    "$(csv_escape "$crit")" "$log" >> "$OUT_CSV"

  if [[ "$SLEEP_SEC" != "0" ]]; then
    sleep "$SLEEP_SEC"
  fi
done