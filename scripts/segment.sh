#!/bin/bash

TRAINSET=train-full-960
TESTSET=dev-other

for VS in 100 ; do
    python segment.py \
        /pio/data/zerospeech2021/quantized/LibriSpeech/${TRAINSET}/quantized_outputs.txt \
        /pio/data/zerospeech2021/quantized/LibriSpeech/${TESTSET}/quantized_outputs.txt \
        $VS \
        /pio/scratch/1/i290956/zs2021/simi/models/segmentations/${TRAINSET}_${TESTSET}_vs${VS}_a$1_test \
        --sentencepiece_prefix=/pio/scratch/1/i290956/zs2021/simi/models/sentencepieces/${TRAINSET}_vs${VS} \
        --segmentation_output_format=csv \
        --viterbi \
        --alpha $1 \
        --clusterings=/pio/scratch/1/i290956/zs2021/clusterings/LibriSpeech/${TESTSET}
done;