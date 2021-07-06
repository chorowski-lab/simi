#!/bin/bash

TRAINSET=train-full-960
TESTSET=dev-clean

# Space-delimited lists
: ${ALPHA:="1.0 2.0 5.0 8.0 10.0 12.0 15.0 20.0"}
: ${VS:="1000 "}

echo "==================================================="
echo "  WARNING: Jobs will be run in parallel."
echo "           Be careful not to overload the system."
echo "==================================================="
sleep 5

for ALPH_ in $ALPHA; do
    for VS_ in $VS ; do
        python segment.py \
            /pio/data/zerospeech2021/quantized/LibriSpeech/${TRAINSET}/quantized_outputs.txt \
            /pio/data/zerospeech2021/quantized/LibriSpeech/${TESTSET}/quantized_outputs.txt \
            $VS_ \
            segmentations/${TRAINSET}_${TESTSET}_vs${VS_}_a${ALPH_}_test \
            --sentencepiece_prefix=/pio/scratch/1/i290956/zs2021/simi/models/sentencepieces/${TRAINSET}_vs${VS_} \
            --segmentation_output_format=csv \
            --viterbi \
            --alpha $ALPH_ \
            --clusterings=/pio/scratch/1/i290956/zs2021/clusterings/LibriSpeech/${TESTSET} &
    done
done
