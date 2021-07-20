#!/bin/bash

source config.sh

: ${VOCAB_SIZE:=100}
: ${ALPHA:=10.0}

python grouping.py \
    /pio/scratch/1/i290956/zs2021/simi/models/segmentations_mpl100/train-clean-100_train-clean-100_vs1000_a10.0/viterbi_segmentation/ \
    /pio/scratch/1/i290956/zs2021/simi/models/segmentations_mpl100/train-clean-100_train-clean-100_vs1000_a10.0/viterbi_segmentation_clustered_${VOCAB_SIZE}/ \
    --word2vec_path=models/word2vec/train-clean-100_vs1000_a${ALPHA}_c${VOCAB_SIZE} \
    --kmeans_path=models/kmeans/train-clean-100_vs1000_a${ALPHA}_c${VOCAB_SIZE} \
    --vocab_size=${VOCAB_SIZE}
