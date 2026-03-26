#!/usr/bin/env bash
source ~/benchmark/bin/activate

cd ~/TensorRTProfiling
PWD=`pwd`
export PYTHONPATH="$PWD"

for map in blocks NH; do
    for format in FP16 INT8; do
        python tensorrtConversion/torch2trt.py --format ${format} --map ${map}
    done
done

for map in blocks NH; do
    for format in FP16 INT8; do
        bash complete_profiling.sh ./ConvertedNNs ${map} 10 10 ${format}
    done
done

for map in blocks NH; do
    for HT in base, FP-TMR, RP-TMR, Ranger, Model1, Model2, Model3, Model4, SelectiveTMR, PredictionFP-TMR,Prediction RP-TMR; do
        python tensorrtConversion/torch2trtHT.py --map ${map} --ht ${HT}
    done
done

for map in blocks NH; do
    for HT in base, FP-TMR, RP-TMR, Ranger, Model1, Model2, Model3, Model4, SelectiveTMR, PredictionFP-TMR,Prediction RP-TMR; do
        bash complete_HT_profiling.sh ./ConvertedNNs ${map} 10 10 ${HT}
    done
done