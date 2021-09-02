#!/bin/bash

source config.sh

SEMANTIC_SUBSET_1=dev           # dev / test
SEMANTIC_SUBSET_2=librispeech   # librispeech / synthetic
VOCAB_SIZE=3000                 # size of the initial vocabulary (for segmentation)
REDUCED_VOCAB_SIZE=200          # reduced vocabulary (at most 500)
TRAINSET=train-clean-100        # trainset


# you probably should not change these variables
TRAINSET_PATH=/pio/data/zerospeech2021/quantized/LibriSpeech/${TRAINSET}/quantized_outputs.txt

TESTSET=semantic-${SEMANTIC_SUBSET_1}-${SEMANTIC_SUBSET_2}
TESTSET_PATH=/pio/data/zerospeech2021/quantized/semantic/${SEMANTIC_SUBSET_1}/${SEMANTIC_SUBSET_2}/quantized_outputs.txt

SENTENCEPIECE_PREFIX=models/sentencepieces/${TRAINSET}_vs${VOCAB_SIZE}
WORD2VEC_PATH=models/word2vec/${TRAINSET}_vs${VOCAB_SIZE}_sp
KMEANS_PATH=models/kmeans/${TRAINSET}_vs${VOCAB_SIZE}_sp/vs${REDUCED_VOCAB_SIZE}

TRAINSET_SEGMENTATION_PATH=models/segmentations/${TRAINSET}_${TRAINSET}_vs${VOCAB_SIZE}/sentencepiece_segmentation
TESTSET_SEGMENTATION_PATH=models/segmentations/${TRAINSET}_${TESTSET}_vs${VOCAB_SIZE}/sentencepiece_segmentation


# train sentencepiece
python train_sentencepiece.py \
    $TRAINSET_PATH \
    $SENTENCEPIECE_PREFIX \
    $VOCAB_SIZE

# eval trainset sentencepiece segmentation
python eval_segmentation.py \
    $TRAINSET_PATH \
    $SENTENCEPIECE_PREFIX \
    $TRAINSET_SEGMENTATION_PATH \
    --output_format=txt

# eval testset sentencepiece segmentation
python eval_segmentation.py \
    $TESTSET_PATH \
    $SENTENCEPIECE_PREFIX \
    $TESTSET_SEGMENTATION_PATH \
    --output_format=txt 

# train word2vec and reduce number of letters in trainset segmentation
python grouping.py \
    $TRAINSET_SEGMENTATION_PATH \
    ${TRAINSET_SEGMENTATION_PATH}/reduced_vs${REDUCED_VOCAB_SIZE} \
    --word2vec_path=${WORD2VEC_PATH} \
    --kmeans_path=${KMEANS_PATH} \
    --segmentation_format=txt \
    --vocab_size=${REDUCED_VOCAB_SIZE}

# reduce number of letters in testset segmentation
python grouping.py \
    $TESTSET_SEGMENTATION_PATH \
    ${TESTSET_SEGMENTATION_PATH}/reduced_vs${REDUCED_VOCAB_SIZE} \
    --word2vec_path=${WORD2VEC_PATH} \
    --kmeans_path=${KMEANS_PATH} \
    --segmentation_format=txt \
    --vocab_size=${REDUCED_VOCAB_SIZE}


echo "Prepared segmentations:"
echo "Train: ${TRAINSET_SEGMENTATION_PATH}/reduced_vs${REDUCED_VOCAB_SIZE}"
echo "Test:  ${TESTSET_SEGMENTATION_PATH}/reduced_vs${REDUCED_VOCAB_SIZE}"