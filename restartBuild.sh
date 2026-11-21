#!/usr/bin/env bash
# Full restart: stop batch → restart Ollama (clear zombie GPU mem) →
# verify CUDA → smoke-test 2 families → resume full build.
# Usage:
#   ./restartBuild.sh              # resume existing knowledge/
#   ./restartBuild.sh --fresh      # wipe knowledge/ + logs/, start over
#   ./restartBuild.sh --limit 2    # extra args passed to build_knowledge.py
set -euo pipefail
cd "$(dirname "$0")"

PY="/home/user/miniforge3/envs/AI/bin/python3"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/batch_knowledge.log"
PID_FILE="$LOG_DIR/batch_knowledge.pid"
SMOKE_LOG="$LOG_DIR/smoke_test.log"
RESUME_FLAG="--resume"
PASSTHRU=()
for a in "$@"; do
  if [ "$a" = "--fresh" ]; then
    RESUME_FLAG=""
  else
    PASSTHRU+=("$a")
  fi
done

echo "=== [1/6] Stop existing batch ==="
if [ -f "$PID_FILE" ]; then
  OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
  if [ -n "${OLD_PID:-}" ]; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$OLD_PID" 2>/dev/null || true
    echo "  killed PID $OLD_PID"
  fi
fi
pkill -f "scripts/build_knowledge.py" 2>/dev/null || true
pkill -f "multiprocessing.spawn" 2>/dev/null || true
sleep 1
if pgrep -f "scripts/build_knowledge.py" >/dev/null; then
  echo "  WARNING: build still running:"
  pgrep -af "scripts/build_knowledge.py" || true
else
  echo "  batch stopped"
fi

echo "=== [2/6] Full Ollama restart (clears zombie holding ~2.8GB GPU) ==="
# Try systemd first (may need password)
if sudo -n systemctl restart ollama 2>/dev/null; then
  echo "  sudo systemctl restart ollama OK"
else
  echo "  sudo needs password — running with your password prompt:"
  sudo systemctl restart ollama
fi
# Belt-and-suspenders: reap any leftover ollama children
pkill -9 -P "$(pgrep -x ollama | head -1)" 2>/dev/null || true
sleep 3

echo "=== [3/6] Verify Ollama + GPU ==="
if curl -s -m 10 http://localhost:11434/api/tags >/dev/null; then
  echo "  ollama /api/tags: OK"
else
  echo "  ERROR: ollama not responding on :11434"
  exit 1
fi

# Zombie / VRAM check
COMPUTE=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
echo "  GPU compute apps:"
if [ -n "$COMPUTE" ]; then
  echo "$COMPUTE" | sed 's/^/    /'
else
  echo "    (none)"
fi
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "  VRAM free: ${FREE_MIB:-?}/${TOTAL_MIB:-?} MiB"
if [ -n "${FREE_MIB:-}" ] && [ "$FREE_MIB" -lt 1500 ]; then
  echo "  ERROR: free VRAM ${FREE_MIB} MiB < 1500 — torch CUDA will segfault."
  echo "  Check: nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
  exit 1
fi

echo "=== [4/6] Verify torch CUDA (must not segfault) ==="
if ! CUDA_VISIBLE_DEVICES=0 "$PY" -c \
  "import torch; assert torch.cuda.is_available(); print('  torch', torch.__version__, '->', torch.cuda.get_device_name(0))"; then
  echo "  ERROR: torch CUDA check failed (segfault or unavailable)"
  exit 1
fi

echo "=== [5/6] Smoke test: 2 families ==="
mkdir -p "$LOG_DIR"
: >"$SMOKE_LOG"
SMOKE_ARGS=(--timeout 300 --min-seqs 10 --limit 2)
if [ -n "$RESUME_FLAG" ]; then
  SMOKE_ARGS+=("$RESUME_FLAG")
fi
set +e
"$PY" scripts/build_knowledge.py "${SMOKE_ARGS[@]}" >"$SMOKE_LOG" 2>&1
SMOKE_RC=$?
set -e
OK_N=$(grep -c "OK:" "$SMOKE_LOG" 2>/dev/null || true)
FAIL_N=$(grep -c "FAILED:" "$SMOKE_LOG" 2>/dev/null || true)
echo "  smoke rc=$SMOKE_RC  OK=$OK_N  FAILED=$FAIL_N  log=$SMOKE_LOG"
if [ "${OK_N:-0}" -lt 1 ]; then
  echo "  ERROR: smoke test produced 0 OK families. Last lines:"
  tail -40 "$SMOKE_LOG" | sed 's/^/    /'
  exit 1
fi
echo "  smoke passed"

echo "=== [6/6] Launch full build ==="
if [ -z "$RESUME_FLAG" ]; then
  echo "  fresh start: wiping knowledge/ and $LOG_DIR/"
  find knowledge -mindepth 1 -delete 2>/dev/null || true
  # keep smoke log path after wipe
  : >"$LOG_FILE"
else
  echo "  resume: keeping existing knowledge/"
fi

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
echo
echo "[restartBuild.sh] full build PID=$PID"
echo "[restartBuild.sh] log:    tail -f $LOG_FILE"
echo "[restartBuild.sh] stop:   kill \$(cat $PID_FILE)"
