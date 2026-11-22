#!/bin/zsh
# GPU job runner: takes the first line of queue.todo ("name<TAB>command"), removes it, runs it, repeats.
# Edit queue.todo to reorder pending jobs; the running job is never touched. One GPU job at a time, always.
cd "${0:A:h}"
export PYTHONWARNINGS=ignore
echo "[qrun] pid $$ start $(date '+%F %T')"
while [ -s queue.todo ]; do
  line=$(head -1 queue.todo); tail -n +2 queue.todo > queue.todo.tmp && mv queue.todo.tmp queue.todo
  name=${line%%$'\t'*}; cmd=${line#*$'\t'}
  echo "[qrun] $name start $(date '+%F %T')"
  sh -c "$cmd"
  echo "[qrun] $name exit $? $(date '+%F %T')"
done
echo "[qrun] done $(date '+%F %T')"
