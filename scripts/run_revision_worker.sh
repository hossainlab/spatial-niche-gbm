#!/usr/bin/env bash
# One worker of the revision benchmark queue (all workers share every job list;
# per-job claim files in 15_revision_benchmark.py prevent duplicate work).
#   UKF=1 -> first drain the UKF pathologist-label queue, then the SNUH queue.
# SNUH order: tuning -> (once all tuning runs exist, exactly one worker runs
# make-final) -> final evaluation -> ablations / marker-gene removal.
# jobs_tune.jsonl / jobs_extra.jsonl are jobs_stage1.jsonl split by phase.
# Usage: UKF=1 nohup caffeinate -i bash scripts/run_revision_worker.sh <id> >> log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=scripts/15_revision_benchmark.py; R=results/revision
run() { "$PY" "$S" run --worker 0 --n-workers 1 --threads 2 "$@"; }

if [ "${UKF:-0}" = 1 ]; then
  run --jobs results/revision_ukf/jobs.jsonl --out results/revision_ukf \
      --data data/preprocessed_histology/ukf --task histology
fi
# tuning first (make-final needs only these), then finals, then ablations
run --jobs "$R/jobs_tune.jsonl" --out "$R"

n_tune=$(wc -l < "$R/jobs_tune.jsonl")
until [ "$(find "$R/runs" -name 'tune_*.json' -o -name 'tune_*.json.failed' | wc -l)" -ge "$n_tune" ]; do sleep 60; done
if mkdir "$R/.make_final_lock" 2>/dev/null; then
  "$PY" "$S" make-final --out "$R" && touch "$R/.final_ready"
fi
until [ -f "$R/.final_ready" ]; do sleep 15; done
run --jobs "$R/jobs_final.jsonl" --out "$R"
run --jobs "$R/jobs_extra.jsonl" --out "$R"
echo "worker $1 done $(date)"
