#!/usr/bin/env bash

ROOT_DIR=$1
map=$2
runs=$3
samples=$4
data_format=$5

COMPLETE_PATH=${ROOT_DIR}/${map}/NN/${data_format}/${HT}
COMPLETE_PATH="$(realpath "$COMPLETE_PATH")"


Opzioni opzionali controllate via env
EXTRA_ARGS=()

plan_path=${COMPLETE_PATH}/NN.plan
out_json=${COMPLETE_PATH}/NN.json
echo "Root path: $COMPLETE_PATH"

trtexec \
  --iterations="${runs}" \
  --loadEngine="${plan_path}" \
  --dumpProfile \
  --exportTimes="${plan_path%%.*}_times.json" \
  --profilingVerbosity=detailed \
  --separateProfileRun

trtexec \
    --loadEngine="$plan_path" \
    --exportLayerInfo="$out_json" \
    --profilingVerbosity=detailed

python trt_benchmarking.py --root ${COMPLETE_PATH} --runs ${runs} --samples ${samples}

python compare.py ${COMPLETE_PATH} -o out_report/${map}/report.csv