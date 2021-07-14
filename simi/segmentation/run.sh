#!/bin/bash

source config.sh

: ${TRAINSET:=train-clean-100}
: ${TESTSET:=train-clean-100}
: ${ALPHA:=10.0}
: ${VOCAB_SIZE:=1000}

python segmentation.py \
    $LIBRISPEECH_QUANTIZED_TRAIN_100 \
    $LIBRISPEECH_QUANTIZED_TRAIN_100 \
    $VOCAB_SIZE \
    models/segmentations_test/${TRAINSET}_${TESTSET}_vs${VOCAB_SIZE}_a${ALPHA} \
    --sentencepiece_prefix=models/sentencepieces_test/${TRAINSET}_vs${VOCAB_SIZE} \
    --segmentation_output_format=csv \
    --alpha=${ALPHA} \
    --clusterings=${LIBRISPEECH_CLUSTERINGS_TRAIN_100} \
    --viterbi
