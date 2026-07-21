#!/usr/bin/env bash
set -Eeuo pipefail

# Manual cyclic runner: run for N hours, rest for M hours, then repeat.
# Override these variables before "start" when a different rhythm is needed.
PROJECT_ROOT="${PROJECT_ROOT:-/home/suati/桌面/国际标准}"
PYTHON_BIN="${PYTHON_BIN:-/home/suati/miniconda3/envs/tuili/bin/python3}"
PIPELINE_MODE="${PIPELINE_MODE:-auto}"
RUN_HOURS="${RUN_HOURS:-7}"
REST_HOURS="${REST_HOURS:-1}"
GRACE_MINUTES="${GRACE_MINUTES:-5}"
RUN_SECONDS="${RUN_SECONDS:-$((RUN_HOURS * 3600))}"
REST_SECONDS="${REST_SECONDS:-$((REST_HOURS * 3600))}"

REPORT_ROOT="${PROJECT_ROOT}/年报"
STATE_DIR="${PROJECT_ROOT}/logs/manual_cycle"
PID_FILE="${STATE_DIR}/runner.pid"
CHILD_PID_FILE="${STATE_DIR}/child.pid"
STOP_FILE="${STATE_DIR}/stop.requested"
PHASE_FILE="${STATE_DIR}/phase.txt"
CONTROL_LOG="${STATE_DIR}/controller.log"
LOCK_FILE="${PROJECT_ROOT}/.measurement-window.lock"
SCRIPT_PATH="$(readlink -f "$0")"

mkdir -p "$STATE_DIR"

log() {
  # The worker's stdout is already appended to CONTROL_LOG by start_runner.
  printf '%s %s\n' "$(date '+%F %T')" "$*"
}

runner_is_alive() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null || true)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

cleanup_worker() {
  rm -f "$PID_FILE" "$CHILD_PID_FILE" "$STOP_FILE"
}

finalize_stage2_year() {
  local year=$1
  local base_dir=$2
  local inference_log=$3
  local status

  set +e
  "$PYTHON_BIN" "${PROJECT_ROOT}/scripts/run_pipeline.py" \
    --config "${PROJECT_ROOT}/configs/pipeline.toml" \
    --project-root "$PROJECT_ROOT" \
    map-main-gb --year "$year" --base-dir "$base_dir" \
    >>"$inference_log" 2>&1
  status=$?
  if (( status == 0 )); then
    "$PYTHON_BIN" "${PROJECT_ROOT}/scripts/run_pipeline.py" \
      --config "${PROJECT_ROOT}/configs/pipeline.toml" \
      --project-root "$PROJECT_ROOT" \
      aggregate-main --year "$year" --base-dir "$base_dir" \
      >>"$inference_log" 2>&1
    status=$?
  fi
  set -e
  return "$status"
}

stage1_is_complete() {
  local text_units=$1
  local stage1_output=$2

  [[ -s "$text_units" && -s "$stage1_output" ]] || return 1
  "$PYTHON_BIN" -c '
import csv
import sys

def read_ids(path, required_columns):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = set(required_columns) - columns
        if missing:
            raise ValueError(f"missing columns in {path}: {sorted(missing)}")
        values = [row["text_unit_id"].strip() for row in reader if row.get("text_unit_id", "").strip()]
    return values

try:
    expected = read_ids(sys.argv[1], {"text_unit_id"})
    completed = read_ids(
        sys.argv[2],
        {"text_unit_id", "relevance", "confidence_score", "reason", "stage1_status", "stage1_error"},
    )
except (OSError, UnicodeError, csv.Error, ValueError):
    raise SystemExit(1)

valid = bool(expected) and len(expected) == len(set(expected)) and len(completed) == len(set(completed))
valid = valid and set(expected) == set(completed)
raise SystemExit(0 if valid else 1)
' "$text_units" "$stage1_output"
}

stage2_input_is_available() {
  local stage2_input=$1
  [[ -s "$stage2_input" ]]
}

select_year_mode() {
  local requested_mode=$1
  local text_units=$2
  local keyword_features=$3
  local stage1_output=$4
  local stage2_input=$5

  if [[ "$requested_mode" != "auto" ]]; then
    printf '%s\n' "$requested_mode"
    return 0
  fi

  if stage2_input_is_available "$stage2_input"; then
    printf '%s\n' "stage2"
  elif [[ -s "$text_units" && -s "$keyword_features" ]]; then
    if stage1_is_complete "$text_units" "$stage1_output"; then
      printf '%s\n' "stage2"
    else
      printf '%s\n' "stage1"
    fi
  else
    printf '%s\n' "full"
  fi
}

run_pipeline_step() {
  local segment_deadline=$1
  local inference_log=$2
  shift 2
  local remaining_seconds status child_pid

  [[ ! -f "$STOP_FILE" ]] || return 130
  remaining_seconds=$((segment_deadline - $(date +%s)))
  (( remaining_seconds > 0 )) || return 124

  timeout --foreground --signal=INT --kill-after="${GRACE_MINUTES}m" "${remaining_seconds}s" \
    "$PYTHON_BIN" "${PROJECT_ROOT}/scripts/run_pipeline.py" \
      --config "${PROJECT_ROOT}/configs/pipeline.toml" \
      --project-root "$PROJECT_ROOT" \
      "$@" \
      >>"$inference_log" 2>&1 &
  child_pid=$!
  echo "$child_pid" >"$CHILD_PID_FILE"
  wait "$child_pid"
  status=$?
  rm -f "$CHILD_PID_FILE"
  return "$status"
}

run_worker() {
  cd "$PROJECT_ROOT"
  echo "$$" >"$PID_FILE"
  trap cleanup_worker EXIT

  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "blocked: another scheduled/manual measurement process holds $LOCK_FILE" >"$PHASE_FILE"
    log "Cannot start: another measurement runner is active."
    return 1
  fi

  case "$PIPELINE_MODE" in
    auto|full|stage1|stage2) ;;
    *)
      echo "invalid PIPELINE_MODE: $PIPELINE_MODE" >"$PHASE_FILE"
      log "PIPELINE_MODE must be 'auto', 'full', 'stage1', or 'stage2'."
      return 2
      ;;
  esac

  log "Manual cycle runner started: mode ${PIPELINE_MODE}, run ${RUN_HOURS}h, rest ${REST_HOURS}h."

  while [[ ! -f "$STOP_FILE" ]]; do
    mapfile -t report_dirs < <(find "$REPORT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
    pending_found=0
    segment_deadline=$(( $(date +%s) + RUN_SECONDS ))
    echo "running: segment ends around $(date -d "@$segment_deadline" '+%F %T')" >"$PHASE_FILE"
    log "Inference segment started; soft limit is ${RUN_HOURS}h."

    for dir in "${report_dirs[@]}"; do
      [[ -f "$STOP_FILE" ]] && break

      name=$(basename "$dir")
      year=${name%%_*}
      case "$year" in
        2024|2025) ;;
        *) continue ;;
      esac

      base_dir="${PROJECT_ROOT}/data/measurement/${year}/main_regression"
      complete_marker="${base_dir}/.pipeline_complete"
      inference_log="${STATE_DIR}/main_regression_${year}_$(date +%F).log"
      text_units="${base_dir}/stage/01_text_units_${year}.csv"
      keyword_features="${base_dir}/stage/02_keyword_features_${year}.csv"
      stage1_output="${base_dir}/stage/03_stage1_llm_relevance_${year}.csv"
      stage2_input="${base_dir}/stage/04_stage2_input_${year}.csv"
      [[ -f "$complete_marker" ]] && continue
      pending_found=1

      remaining_seconds=$((segment_deadline - $(date +%s)))
      (( remaining_seconds > 0 )) || break

      selected_mode=$(select_year_mode \
        "$PIPELINE_MODE" "$text_units" "$keyword_features" "$stage1_output" "$stage2_input")
      log "Starting/resuming ${year} from ${selected_mode}; ${remaining_seconds}s remain in this segment."
      echo "running: ${year} from ${selected_mode}, segment ends around $(date -d "@$segment_deadline" '+%F %T')" >"$PHASE_FILE"

      status=0
      set +e
      case "$selected_mode" in
        full)
          run_pipeline_step "$segment_deadline" "$inference_log" \
            main-regression \
            --year "$year" \
            --input-dir "$dir" \
            --provider vllm_batch \
            --base-dir "$base_dir"
          status=$?
          ;;
        stage1)
          if [[ ! -s "$text_units" || ! -s "$keyword_features" ]]; then
            log "Cannot start from stage1: text units or keyword features are missing for ${year}."
            status=2
          else
            run_pipeline_step "$segment_deadline" "$inference_log" \
              stage1-screen \
              --year "$year" \
              --provider vllm_batch \
              --base-dir "$base_dir"
            status=$?
            if (( status == 0 )); then
              run_pipeline_step "$segment_deadline" "$inference_log" \
                route-main \
                --year "$year" \
                --base-dir "$base_dir"
              status=$?
            fi
            if (( status == 0 )); then
              run_pipeline_step "$segment_deadline" "$inference_log" \
                stage2-extract \
                --year "$year" \
                --provider vllm_batch \
                --base-dir "$base_dir"
              status=$?
            fi
          fi
          ;;
        stage2)
          if [[ ! -s "$text_units" ]]; then
            log "Cannot start from stage2: missing text units ${text_units}, required for final aggregation."
            status=2
          elif ! stage2_input_is_available "$stage2_input"; then
            if [[ ! -s "$keyword_features" ]] || ! stage1_is_complete "$text_units" "$stage1_output"; then
              log "Cannot start from stage2: stage2 input is missing and stage1 is not complete for ${year}."
              status=2
            else
              log "Stage1 is complete; building the missing stage2 route input for ${year}."
              run_pipeline_step "$segment_deadline" "$inference_log" \
                route-main \
                --year "$year" \
                --base-dir "$base_dir"
              status=$?
            fi
          fi
          if (( status == 0 )); then
            run_pipeline_step "$segment_deadline" "$inference_log" \
              stage2-extract \
              --year "$year" \
              --provider vllm_batch \
              --base-dir "$base_dir"
            status=$?
          fi
          ;;
      esac
      set -e

      if [[ -f "$STOP_FILE" ]]; then
        log "Manual stop completed after checkpointing the active chunk."
        echo "stopped: manual request at $(date '+%F %T')" >"$PHASE_FILE"
        return 0
      fi

      case "$status" in
        0)
          if [[ "$selected_mode" != "full" ]]; then
            log "${year} LLM stages completed; running GB mapping and aggregation."
            if ! finalize_stage2_year "$year" "$base_dir" "$inference_log"; then
              log "${year} post-processing failed; inspect ${inference_log}."
              echo "failed: ${year} post-processing, $(date '+%F %T')" >"$PHASE_FILE"
              return 1
            fi
          fi
          touch "$complete_marker"
          log "${year} completed successfully."
          ;;
        124|130)
          log "Inference segment ended; ${year} will resume after the rest period."
          break
          ;;
        *)
          log "${year} failed with exit code ${status}; inspect ${inference_log}."
          echo "failed: ${year}, exit ${status}, $(date '+%F %T')" >"$PHASE_FILE"
          return "$status"
          ;;
      esac
    done

    [[ -f "$STOP_FILE" ]] && continue

    # Recheck after the segment because the last pending year may just have completed.
    still_pending=0
    for dir in "${report_dirs[@]}"; do
      year=$(basename "$dir")
      year=${year%%_*}
      case "$year" in
        2024|2025)
          [[ -f "${PROJECT_ROOT}/data/measurement/${year}/main_regression/.pipeline_complete" ]] \
            || still_pending=1
          ;;
      esac
    done

    if (( pending_found == 0 || still_pending == 0 )); then
      log "All available 2024/2025 work completed; runner is stopping automatically."
      echo "completed: all available work at $(date '+%F %T')" >"$PHASE_FILE"
      return 0
    fi

    rest_deadline=$(( $(date +%s) + REST_SECONDS ))
    echo "resting: resumes around $(date -d "@$rest_deadline" '+%F %T')" >"$PHASE_FILE"
    log "Rest period started for ${REST_HOURS}h."
    while (( $(date +%s) < rest_deadline )); do
      if [[ -f "$STOP_FILE" ]]; then
        log "Manual stop completed during the rest period."
        echo "stopped: manual request at $(date '+%F %T')" >"$PHASE_FILE"
        return 0
      fi
      sleep 5
    done
  done

  echo "stopped: manual request at $(date '+%F %T')" >"$PHASE_FILE"
}

start_runner() {
  if runner_is_alive; then
    echo "Runner is already active (PID $(cat "$PID_FILE"))."
    exit 0
  fi

  rm -f "$PID_FILE" "$CHILD_PID_FILE" "$STOP_FILE"
  export PROJECT_ROOT PYTHON_BIN PIPELINE_MODE RUN_HOURS REST_HOURS GRACE_MINUTES RUN_SECONDS REST_SECONDS
  nohup "$SCRIPT_PATH" _run >>"$CONTROL_LOG" 2>&1 &
  runner_pid=$!
  echo "$runner_pid" >"$PID_FILE"
  sleep 1

  if kill -0 "$runner_pid" 2>/dev/null; then
    echo "Runner started in the background (PID $runner_pid)."
    echo "Use '$SCRIPT_PATH status' to inspect it."
  else
    echo "Runner did not stay active. Inspect: $CONTROL_LOG"
    tail -n 30 "$CONTROL_LOG" || true
    exit 1
  fi
}

parse_start_args() {
  while (( $# > 0 )); do
    case "$1" in
      --from|--from-stage)
        if (( $# < 2 )); then
          echo "Missing value after $1." >&2
          return 2
        fi
        PIPELINE_MODE=$2
        shift 2
        ;;
      --from=*|--from-stage=*)
        PIPELINE_MODE=${1#*=}
        shift
        ;;
      *)
        echo "Unknown start option: $1" >&2
        return 2
        ;;
    esac
  done

  case "$PIPELINE_MODE" in
    auto|full|stage1|stage2) ;;
    *)
      echo "--from must be one of: auto, full, stage1, stage2." >&2
      return 2
      ;;
  esac
}

stop_runner() {
  if ! runner_is_alive; then
    echo "Runner is not active."
    rm -f "$PID_FILE" "$CHILD_PID_FILE" "$STOP_FILE"
    exit 0
  fi

  touch "$STOP_FILE"
  if [[ -f "$CHILD_PID_FILE" ]]; then
    child_pid=$(cat "$CHILD_PID_FILE" 2>/dev/null || true)
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
      kill -INT "$child_pid" 2>/dev/null || true
    fi
  fi
  echo "Stop requested. The active vLLM chunk will be checkpointed before exit."
  echo "Use '$SCRIPT_PATH status' to confirm it has stopped."
}

show_status() {
  if runner_is_alive; then
    echo "Runner: active (PID $(cat "$PID_FILE"))"
  else
    echo "Runner: inactive"
  fi
  if [[ -f "$PHASE_FILE" ]]; then
    echo "Phase: $(cat "$PHASE_FILE")"
  fi
  echo "Controller log: $CONTROL_LOG"
}

show_usage() {
  cat <<EOF
Usage: $SCRIPT_PATH start [--from auto|full|stage1|stage2]
       $SCRIPT_PATH {stop|status|logs}

Examples:
  $SCRIPT_PATH start
  $SCRIPT_PATH start --from stage1
  $SCRIPT_PATH start --from stage2
  $SCRIPT_PATH start --from full
  $SCRIPT_PATH stop
  $SCRIPT_PATH status
  $SCRIPT_PATH logs

Start modes:
  auto    Reuse the latest complete intermediate files (default).
  full    Rebuild text units and keyword features before LLM stages.
  stage1  Reuse text units/keywords; resume from Stage 1 LLM screening.
  stage2  Reuse Stage 2 input, or build it from a complete Stage 1 result.

Change cycle lengths only when starting a new runner:
  RUN_HOURS=6 REST_HOURS=2 $SCRIPT_PATH start

The PIPELINE_MODE environment variable remains supported:
  PIPELINE_MODE=stage2 $SCRIPT_PATH start
EOF
}

case "${1:-}" in
  start)
    shift
    parse_start_args "$@" || { show_usage; exit 2; }
    start_runner
    ;;
  stop) stop_runner ;;
  status) show_status ;;
  logs) tail -F "$CONTROL_LOG" ;;
  _run) run_worker ;;
  *) show_usage; exit 2 ;;
esac
