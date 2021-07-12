#!/bin/bash

: ${VS:="10000 20000 50000 "}
: ${VVS:="50 75 100 200 300 400"}
: ${ALPHA:=10.0}

for VS_ in $VS; do
    for VVS_ in $VVS; do

        python grouping.py \
            /pio/scratch/1/i290956/zs2021/simi/models/segmentations_mpl100/train-clean-100_train-clean-100_vs${VS_}_a${ALPHA}/viterbi_segmentation/ \
            /pio/scratch/1/i290956/zs2021/simi/models/segmentations_mpl100/train-clean-100_train-clean-100_vs${VS_}_a${ALPHA}/viterbi_segmentation_clustered_${VVS_}/ \
            --word2vec_path /pio/scratch/1/i290956/zs2021/simi/models/word2vec_mpl100_sp/train-clean-100_vs${VS_}_a${ALPHA}_c${VVS_} \
            --kmeans_path /pio/scratch/1/i290956/zs2021/simi/models/kmeans_mpl100_sp/train-clean-100_vs${VS_}_a${ALPHA}_c${VVS_}_cosine \
            --vocab_size $VVS_ &
    done;
done;
