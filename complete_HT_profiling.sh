#!/usr/bin/env bash
ROOT_DIR=$1
map=$2
runs=$3
samples=$4
HT=$5

COMPLETE_PATH=${ROOT_DIR}/${map}/HT/${HT}
COMPLETE_PATH="$(realpath "$COMPLETE_PATH")"
echo "Root path: $COMPLETE_PATH"

Opzioni opzionali controllate via env
EXTRA_ARGS=()

echo $COMPLETE_PATH
plan_path=${COMPLETE_PATH}/NN.plan
out_json=${COMPLETE_PATH}/NN.json

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

plan_path=${COMPLETE_PATH}/last.plan
out_json=${COMPLETE_PATH}/NN.json

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

python trtHT_benchmarking.py --root ${ROOT_DIR} --runs ${runs} --samples ${samples} --ht ${HT} --map ${map}

python compare.py ${COMPLETE_PATH} -o out_report/${map}/HTreport.csv