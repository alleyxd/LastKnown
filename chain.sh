#!/bin/bash
# wait for the tidemark pass of run_all.sh, stop it, then run the rest
cd "$(dirname "$0")"
until [ "$(wc -l < results/qwen3-8b/tidemark.jsonl)" -ge 400 ] || grep -q "== noguard" results/run_all.log; do sleep 60; done
for pid in $(ps -ef | grep "run_all.sh" | grep -v grep | awk '{print $2}'); do kill $pid; done
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*stale.py --method*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
sleep 5
./run_rest.sh > results/run_rest.log 2>&1
