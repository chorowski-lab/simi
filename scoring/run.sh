#!/bin/bash

: ${SUBSET:="train-clean-100"}
: ${VS:="1000 "}
: ${ALPHA:=10.0}
: ${FRAME_SHIFT:=0}
: ${CLUSTERING_SIZE:="50 75 100 200 300 400 "}


# for VS_ in $VS; do

#     echo "Scoring LibriSpeech ${SUBSET} vocab size ${VS_}"

#     python scoring/simple_score.py \
#         /pio/data/zerospeech2021/librispeech_alignments/$SUBSET \
#         /pio/scratch/1/i290956/zs2021/simi/models/segmentations_mpl100/train-clean-100_${SUBSET}_vs${VS_}_a${ALPHA}/sentencepiece_segmentation \
#         $FRAME_SHIFT
        
#     python scoring/simple_score.py \
#         /pio/data/zerospeech2021/librispeech_alignments/$SUBSET \
#         /pio/scratch/1/i290956/zs2021/simi/models/segmentations_mpl100/train-clean-100_${SUBSET}_vs${VS_}_a${ALPHA}/viterbi_segmentation \
#         $FRAME_SHIFT
# done

for CS in $CLUSTERING_SIZE; do

    echo "Scoring LibriSpeech ${SUBSET} clustering size ${CS}"

    python scoring/simple_score.py \
        /pio/data/zerospeech2021/librispeech_alignments/$SUBSET \
        /pio/scratch/1/i290956/zs2021/simi/models/segmentations_mpl100/train-clean-100_${SUBSET}_vs1000_a${ALPHA}/viterbi_segmentation_clustered_${CS} \
        $FRAME_SHIFT
        
done
