#!/bin/bash

source config.sh

VOCAB_SIZE=50000    # size of the vocabulary

TRAINSET_PATH=models/segmentations/train-clean-100_train-clean-100_vs3000/sentencepiece_segmentation/reduced_vs200/clustered_outputs_200.txt
TESTSET_LIBRISPEECH_PATH=models/segmentations/train-clean-100_semantic-dev-librispeech_vs3000/sentencepiece_segmentation/reduced_vs200/clustered_outputs_200.txt
TESTSET_SYNTHETIC_PATH=models/segmentations/train-clean-100_semantic-dev-synthetic_vs3000/sentencepiece_segmentation/reduced_vs200/clustered_outputs_200.txt
OUTPUT_PATH=submission/semantic/dev


RUN_ID=`echo -n ${TRAINSET_PATH}${TESTSET_LIBRISPEECH_PATH}${TESTSET_SYNTHETIC_PATH}${VOCAB_SIZE} | md5sum | cut -f1 -d" "`

TRAINSET_SEGMENTATION_PATH=tmp/simi_task/${RUN_ID}/segmentation
SENTENCEPIECE_PREFIX=tmp/simi_task/${RUN_ID}/sentencepiece
WORD2VEC_PATH=tmp/simi_task/${RUN_ID}/word2vec
TESTSET_STRINGIFIED_LIBRISPEECH_PATH=tmp/simi_task/${RUN_ID}/testset_strigified_librispeech.txt
TESTSET_STRINGIFIED_SYNTHETIC_PATH=tmp/simi_task/${RUN_ID}/testset_strigified_synthetic.txt


# train sentencepiece segmentation
python train_sentencepiece.py \
    $TRAINSET_PATH \
    $SENTENCEPIECE_PREFIX \
    $VOCAB_SIZE \
    --max_piece_length=16


# eval trainset segmentation
python eval_segmentation.py \
    $TRAINSET_PATH \
    $SENTENCEPIECE_PREFIX \
    $TRAINSET_SEGMENTATION_PATH \
    --output_format=txt 


# stringify both testsets
python simi/stringify.py \
    $TESTSET_LIBRISPEECH_PATH \
    $TESTSET_STRINGIFIED_LIBRISPEECH_PATH

python simi/stringify.py \
    $TESTSET_SYNTHETIC_PATH \
    $TESTSET_STRINGIFIED_SYNTHETIC_PATH


# compute the semantic vectors
python task.py \
    ${TRAINSET_SEGMENTATION_PATH}/segmented_outputs.txt \
    $TESTSET_STRINGIFIED_LIBRISPEECH_PATH \
    $WORD2VEC_PATH \
    ${OUTPUT_PATH}/librispeech

python task.py \
    ${TRAINSET_SEGMENTATION_PATH}/segmented_outputs.txt \
    $TESTSET_STRINGIFIED_SYNTHETIC_PATH \
    $WORD2VEC_PATH \
    ${OUTPUT_PATH}/synthetic
