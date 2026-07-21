#!/usr/bin/env bash
set -Eeuo pipefail

# These can be overridden from cron or the shell when the remote paths differ.
PROJECT_ROOT="${PROJECT_ROOT:-/home/suati/桌面/国际标准}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WINDOW_START="${WINDOW_START:-00:00}"
WINDOW_END="${WINDOW_END:-10:00}"
GRACE_MINUTES="${GRACE_MINUTES:-5}"
REPORT_ROOT="${PROJECT_ROOT}/年报"
LOG_DIR="${PROJECT_ROOT}/logs/scheduler"
LOCK_FILE="${PROJECT_ROOT}/.measurement-window.lock"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '%s Another scheduled pipeline is already running; exiting.\n' "$(date '+%F %T')"
  exit 0
fi

# Work out the next hard boundary. This supports both same-day windows such as
# 00:00-10:00 and cross-midnight windows such as 23:00-10:00.
now_epoch=$(date +%s)
today=$(date +%F)
start_today_epoch=$(date -d "$today $WINDOW_START" +%s)
end_today_epoch=$(date -d "$today $WINDOW_END" +%s)

if (( start_today_epoch < end_today_epoch )); then
  if (( now_epoch < start_today_epoch || now_epoch >= end_today_epoch )); then
    printf '%s Outside the %s-%s inference window; exiting.\n' \
      "$(date '+%F %T')" "$WINDOW_START" "$WINDOW_END"
    exit 0
  fi
  hard_stop_epoch=$end_today_epoch
else
  if (( now_epoch >= start_today_epoch )); then
    hard_stop_epoch=$(date -d "tomorrow $WINDOW_END" +%s)
  elif (( now_epoch < end_today_epoch )); then
    hard_stop_epoch=$end_today_epoch
  else
    printf '%s Outside the %s-%s inference window; exiting.\n' \
      "$(date '+%F %T')" "$WINDOW_START" "$WINDOW_END"
    exit 0
  fi
fi

soft_stop_epoch=$((hard_stop_epoch - GRACE_MINUTES * 60))
if (( now_epoch >= soft_stop_epoch )); then
  printf '%s Less than %s minutes remain before %s; exiting.\n' \
    "$(date '+%F %T')" "$GRACE_MINUTES" "$WINDOW_END"
  exit 0
fi

mapfile -t report_dirs < <(find "$REPORT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

for dir in "${report_dirs[@]}"; do
  name=$(basename "$dir")
  year=${name%%_*}
  case "$year" in
    2024|2025) ;;
    *) continue ;;
  esac

  base_dir="${PROJECT_ROOT}/data/measurement/${year}/main_regression"
  complete_marker="${base_dir}/.pipeline_complete"
  daily_log="${LOG_DIR}/main_regression_${year}_$(date +%F).log"

  if [[ -f "$complete_marker" ]]; then
    printf '%s %s is already complete; skipping.\n' "$(date '+%F %T')" "$year" | tee -a "$daily_log"
    continue
  fi

  now_epoch=$(date +%s)
  remaining_seconds=$((soft_stop_epoch - now_epoch))
  if (( remaining_seconds <= 0 )); then
    printf '%s Daily inference window ended; resume tomorrow.\n' "$(date '+%F %T')" | tee -a "$daily_log"
    exit 0
  fi

  printf '%s Starting/resuming %s from %s.\n' "$(date '+%F %T')" "$year" "$dir" | tee -a "$daily_log"
  set +e
  timeout --foreground --signal=INT --kill-after="${GRACE_MINUTES}m" "${remaining_seconds}s" \
    "$PYTHON_BIN" "${PROJECT_ROOT}/scripts/run_pipeline.py" \
      --config "${PROJECT_ROOT}/configs/pipeline.toml" \
      --project-root "$PROJECT_ROOT" \
      main-regression \
      --year "$year" \
      --input-dir "$dir" \
      --provider vllm_batch \
      --base-dir "$base_dir" \
      >>"$daily_log" 2>&1
  status=$?
  set -e

  case "$status" in
    0)
      touch "$complete_marker"
      printf '%s %s completed successfully.\n' "$(date '+%F %T')" "$year" | tee -a "$daily_log"
      ;;
    124|130)
      printf '%s %s paused at the daily boundary; checkpoints will resume tomorrow.\n' \
        "$(date '+%F %T')" "$year" | tee -a "$daily_log"
      exit 0
      ;;
    *)
      printf '%s %s failed with exit code %s; inspect %s.\n' \
        "$(date '+%F %T')" "$year" "$status" "$daily_log" | tee -a "$daily_log"
      exit "$status"
      ;;
  esac
done

printf '%s No pending 2024/2025 report directory remains.\n' "$(date '+%F %T')"
