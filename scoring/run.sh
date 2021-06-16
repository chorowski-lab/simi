#!/bin/bash

: ${SUBSET:="dev-clean"}
: ${VS:=1000}
: ${ALPHA:=10.0}
: ${BASELINE:=false}
: ${FRAME_SHIFT:=0}

echo "Scoring LibriSpeech ${SUBSET}..."

SUBDIR=viterbi_segmentation
[ "$BASELINE" = true ] && SUBDIR=sentencepiece_segmentation

for ALPH_ in $ALPHA; do

    python scoring/simple_score.py \
        /pio/data/zerospeech2021/librispeech_alignments/$SUBSET \
        /pio/scratch/1/alan/replearn/grzesiek_simi/segmentations/train-full-960_${SUBSET}_vs${VS}_a${ALPH_}/${SUBDIR} \
        $FRAME_SHIFT &
done
