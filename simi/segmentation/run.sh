#!/bin/bash

source config.sh

: ${TRAINSET:=train-clean-100}
: ${TESTSET:=train-clean-100}
: ${ALPHA:=10.0}
: ${VOCAB_SIZE:=1200}

# python segmentation.py \
#     $LIBRISPEECH_QUANTIZED_TRAIN_100 \
#     models/segmentations/${TRAINSET}_${TESTSET}_vs${VOCAB_SIZE}_a${ALPHA}/viterbi_segmentation \
#     --trainset=${LIBRISPEECH_QUANTIZED_TRAIN_100} \
#     --vocab_size=${VOCAB_SIZE} \
#     --sentencepiece_prefix=models/sentencepieces/${TRAINSET}_vs${VOCAB_SIZE} \
#     --output_format=csv \
#     --alpha=${ALPHA} \
#     --clusterings=${LIBRISPEECH_CLUSTERINGS_TRAIN_100} \
#     --viterbi

python train_sentencepiece.py \
    $LIBRISPEECH_QUANTIZED_TRAIN_100 \
    models/sentencepieces/${TRAINSET}_vs${VOCAB_SIZE} \
    $VOCAB_SIZE

python eval_segmentation.py \
    $LIBRISPEECH_QUANTIZED_TRAIN_100 \
    models/sentencepieces/${TRAINSET}_vs${VOCAB_SIZE} \
    models/segmentations/${TRAINSET}_${TESTSET}_vs${VOCAB_SIZE}_a${ALPHA} \
    --output_format=csv,txt \
    --clusterings=${LIBRISPEECH_CLUSTERINGS_TRAIN_100} \
    --viterbi \
    --alpha=${ALPHA}
