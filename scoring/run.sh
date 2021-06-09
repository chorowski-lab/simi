#!/bin/bash

: ${SUBSET:="dev-clean"}

echo "==========================================="
echo "Subset: ${SUBSET}"
echo "==========================================="
echo ""

echo ""
echo "CPC (shift -1, test the scoring function) ======================"
echo ""

python simple_score.py \
    /pio/data/zerospeech2021/librispeech_alignments/$SUBSET \
    /pio/scratch/1/alan/replearn/zerospeech2021_baseline/quantized/$SUBSET/quantized_outputs.txt \
    -1

echo ""


echo "CPC (baseline) =============================="
python fit.py --dataset /pio/data/zerospeech2021/LibriSpeech-wav/$SUBSET \
              --alignments /pio/data/zerospeech2021/librispeech_alignments/$SUBSET \
              --cpc_clusterings /pio/scratch/1/alan/replearn/zerospeech2021_baseline/quantized/$SUBSET/quantized_outputs.txt \

for SHIFT_ in -3 -2 -1 1 2 3 ; do

echo ""
echo "CPC (shift $SHIFT_) =============================="
python fit.py --dataset /pio/data/zerospeech2021/LibriSpeech-wav/$SUBSET \
              --alignments /pio/data/zerospeech2021/librispeech_alignments/$SUBSET \
              --cpc_clusterings /pio/scratch/1/alan/replearn/zerospeech2021_baseline/quantized/$SUBSET/quantized_outputs.txt \
              --shift $SHIFT_

done

###########
