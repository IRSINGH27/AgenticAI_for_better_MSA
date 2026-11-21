#!/usr/bin/env bash
# Fresh knowledge-base build (wipes knowledge/ + logs/, then starts batch).
# Usage:
#   ./build.sh              # wipe + start from scratch (default)
#   ./build.sh --resume     # keep existing KB, resume where left off
#   ./build.sh --limit 50   # extra args passed through to build_knowledge.py
set -euo pipefail
cd "$(dirname "$0")"

PY="/home/user/miniforge3/envs/AI/bin/python3"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/batch_knowledge.log"
PID_FILE="$LOG_DIR/batch_knowledge.pid"
RESUME=0
PASSTHRU=()
for a in "$@"; do
  if [ "$a" = "--resume" ]; then
    RESUME=1
  else
    PASSTHRU+=("$a")
  fi
done

mkdir -p "$LOG_DIR" knowledge

if [ "$RESUME" -eq 0 ]; then
  echo "[build.sh] fresh start: wiping knowledge/ and $LOG_DIR/"
  find knowledge -mindepth 1 -delete 2>/dev/null || true
  find "$LOG_DIR" -mindepth 1 -delete 2>/dev/null || true
  rm -f build.pid
  RESUME_FLAG=""
else
  echo "[build.sh] resume: keeping existing knowledge/"
  RESUME_FLAG="--resume"
fi

# Optional Ollama health check (uncomment to auto-restart):
# sudo systemctl restart ollama
# curl -s -m 30 http://localhost:11434/api/generate \
#   -d '{"model":"llama3","prompt":"say OK","stream":false,"options":{"num_predict":5}}'

ARGS=(--timeout 300 --min-seqs 10)
if [ -n "$RESUME_FLAG" ]; then
  ARGS+=("$RESUME_FLAG")
fi
if [ ${#PASSTHRU[@]} -gt 0 ]; then
  ARGS+=("${PASSTHRU[@]}")
fi

nohup "$PY" scripts/build_knowledge.py "${ARGS[@]}" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"
echo "$PID" >build.pid
echo "[build.sh] started PID=$PID"
echo "[build.sh] log:    $LOG_FILE"
echo "[build.sh] pid:    $PID_FILE (also build.pid)"
echo "[build.sh] follow: tail -f $LOG_FILE"
echo "[build.sh] stop:   kill \$(cat $PID_FILE)"
