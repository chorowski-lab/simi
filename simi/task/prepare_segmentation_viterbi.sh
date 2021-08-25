#!/bin/bash

source config.sh

SEMANTIC_SUBSET_1=dev           # dev / test
SEMANTIC_SUBSET_2=librispeech   # librispeech / synthetic
VOCAB_SIZE=1000
REDUCED_VOCAB_SIZE=100
ALPHA=10.0
TRAINSET=train-clean-100


# you probably should not change these variables
TRAINSET_PATH=/pio/data/zerospeech2021/quantized/LibriSpeech/${TRAINSET}/quantized_outputs.txt
TRAINSET_CLUSTERINGS=/pio/scratch/1/i290956/zs2021/clusterings/LibriSpeech/${TRAINSET}

TESTSET=semantic-${SEMANTIC_SUBSET_1}-${SEMANTIC_SUBSET_2}
TESTSET_PATH=/pio/data/zerospeech2021/quantized/semantic/${SEMANTIC_SUBSET_1}/${SEMANTIC_SUBSET_2}/quantized_outputs.txt
TESTSET_CLUSTERINGS=/pio/scratch/1/i290956/zs2021/clusterings/semantic/${SEMANTIC_SUBSET_1}/${SEMANTIC_SUBSET_2}

SENTENCEPIECE_PREFIX=models/sentencepieces/${TRAINSET}_vs${VOCAB_SIZE}
WORD2VEC_PATH=models/word2vec/${TRAINSET}_vs${VOCAB_SIZE}_viterbi_a${ALPHA}
KMEANS_PATH=models/kmeans/${TRAINSET}_vs${VOCAB_SIZE}_ls${REDUCED_VOCAB_SIZE}_viterbi_a${ALPHA}

TRAINSET_SEGMENTATION_PATH=models/segmentations/${TRAINSET}_${TRAINSET}_vs${VOCAB_SIZE}/viterbi_segmentation_a${ALPHA}
TESTSET_SEGMENTATION_PATH=models/segmentations/${TRAINSET}_${TESTSET}_vs${VOCAB_SIZE}/viterbi_segmentation_a${ALPHA}


# train sentencepiece model
python train_sentencepiece.py \
    $TRAINSET_PATH \
    $SENTENCEPIECE_PREFIX \
    $VOCAB_SIZE

# # eval trainset segmentation
python eval_segmentation.py \
    $TRAINSET_PATH \
    $SENTENCEPIECE_PREFIX \
    $TRAINSET_SEGMENTATION_PATH \
    --output_format=txt \
    --clusterings=${TRAINSET_CLUSTERINGS} \
    --viterbi \
    --alpha=${ALPHA}

# eval testset segmentation
python eval_segmentation.py \
    $TESTSET_PATH \
    $SENTENCEPIECE_PREFIX \
    $TESTSET_SEGMENTATION_PATH \
    --output_format=txt \
    --clusterings=${TESTSET_CLUSTERINGS} \
    --viterbi \
    --alpha=${ALPHA}
    
# reduce number of letters
python grouping.py \
    $TRAINSET_SEGMENTATION_PATH \
    ${TRAINSET_SEGMENTATION_PATH}/reduced_vs${REDUCED_VOCAB_SIZE} \
    --word2vec_path=${WORD2VEC_PATH} \
    --kmeans_path=${KMEANS_PATH} \
    --segmentation_format=txt \
    --vocab_size=${REDUCED_VOCAB_SIZE}

python grouping.py \
    $TESTSET_SEGMENTATION_PATH \
    ${TESTSET_SEGMENTATION_PATH}/reduced_vs${REDUCED_VOCAB_SIZE} \
    --word2vec_path=${WORD2VEC_PATH} \
    --kmeans_path=${KMEANS_PATH} \
    --segmentation_format=txt \
    --vocab_size=${REDUCED_VOCAB_SIZE} \
    --eval


echo "Prepared segmentations:"
echo "Train: ${TRAINSET_SEGMENTATION_PATH}/reduced_vs${REDUCED_VOCAB_SIZE}"
echo "Test:  ${TESTSET_SEGMENTATION_PATH}/reduced_vs${REDUCED_VOCAB_SIZE}"
