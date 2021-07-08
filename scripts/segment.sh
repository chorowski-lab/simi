#!/bin/bash

TRAINSET=train-clean-100
TESTSET=train-clean-100
: ${ALPHA:=10.0}

# Space-delimited lists
: ${VS:="1000 2000 5000 10000 20000 50000"}

echo "==================================================="
echo "  WARNING: Jobs will be run in parallel."
echo "           Be careful not to overload the system."
echo "==================================================="
sleep 1

for VS_ in $VS ; do
    python segment.py \
        /pio/data/zerospeech2021/quantized/LibriSpeech/${TRAINSET}/quantized_outputs.txt \
        /pio/data/zerospeech2021/quantized/LibriSpeech/${TESTSET}/quantized_outputs.txt \
        $VS_ \
        models/segmentations_mpl100/${TRAINSET}_${TESTSET}_vs${VS_}_a${ALPHA} \
        --sentencepiece_prefix=/pio/scratch/1/i290956/zs2021/simi/models/sentencepieces_mpl100/${TRAINSET}_vs${VS_} \
        --segmentation_output_format=csv \
        --alpha=${ALPHA} \
        --clusterings=/pio/scratch/1/i290956/zs2021/clusterings/LibriSpeech/${TESTSET} \
        --viterbi &
done ;
