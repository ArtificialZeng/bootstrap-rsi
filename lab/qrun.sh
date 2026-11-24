#!/bin/zsh
# GPU job runner: takes the first line of queue.todo ("name<TAB>command"), removes it, runs it, repeats.
# Edit queue.todo to reorder pending jobs; the running job is never touched. One GPU job at a time, always.
# Halts (leaving the rest of the queue in place) when a job fails within 120 s or a "-smoke" job fails:
# on 2026-11-25 a runner that had outlived an app restart could no longer reach the Metal compiler and
# every queued job failed in seconds.
cd "${0:A:h}"
export PYTHONWARNINGS=ignore
echo "[qrun] pid $$ start $(date '+%F %T')"
while [ -s queue.todo ]; do
  line=$(head -1 queue.todo); tail -n +2 queue.todo > queue.todo.tmp && mv queue.todo.tmp queue.todo
  name=${line%%$'\t'*}; cmd=${line#*$'\t'}
  echo "[qrun] $name start $(date '+%F %T')"
  t0=$(date +%s)
  sh -c "$cmd"; rc=$?
  echo "[qrun] $name exit $rc $(date '+%F %T')"
  if [ $rc -ne 0 ] && { [ $(( $(date +%s) - t0 )) -lt 120 ] || [[ $name == *-smoke ]]; }; then
    { print -r -- "$line"; cat queue.todo; } > queue.todo.tmp && mv queue.todo.tmp queue.todo
    echo "[qrun] halted after $name (exit $rc); queue kept $(date '+%F %T')"; exit 1
  fi
done
echo "[qrun] done $(date '+%F %T')"
