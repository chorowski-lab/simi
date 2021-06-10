#!/bin/bash

TRAINSET=train-full-960
TESTSET=dev-other

# : ${ALPHA:=10.0}

: ${VS:=1000}

for ALPHA in 1.0 2.0 5.0 8.0 10.0 12.0 15.0 20.0; do
    # for VS in 1000 ; do
        python segment.py \
            /pio/data/zerospeech2021/quantized/LibriSpeech/${TRAINSET}/quantized_outputs.txt \
            /pio/data/zerospeech2021/quantized/LibriSpeech/${TESTSET}/quantized_outputs.txt \
            $VS \
            segmentations/${TRAINSET}_${TESTSET}_vs${VS}_a${ALPHA}_test \
            --sentencepiece_prefix=/pio/scratch/1/i290956/zs2021/simi/models/sentencepieces/${TRAINSET}_vs${VS} \
            --segmentation_output_format=csv \
            --viterbi \
            --alpha $ALPHA \
            --clusterings=/pio/scratch/1/i290956/zs2021/clusterings/LibriSpeech/${TESTSET} &
    # done
done
