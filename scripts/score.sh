#!/bin/bash




# python ./scoring/fit.py \
#     --dataset /pio/data/zerospeech2021/LibriSpeech-wav/dev-clean \
#     --alignments  /pio/data/zerospeech2021/librispeech_alignments/dev-clean \
    # --cpc_clusterings /pio/scratch/1/i290956/zs2021/simi/results/dev-clean_vs500_w2v100_ncl50/quantized_outputs.txt

for SEGMENTATION in sentencepiece viterbi ; do

    for SHIFT_ in -3 -2 -1 0 1 2 3 ; do

    echo ""
    echo "segmentation: $SEGMENTATION shift: $SHIFT_ =============================="
    python ./scoring/simple_score.py /pio/scratch/1/i290956/zs2021/simi/results/train-clean-100_dev-clean_vs500/${SEGMENTATION}_segmentation \
                    /pio/data/zerospeech2021/quantized/LibriSpeech/dev-clean/quantized_outputs.txt $SHIFT_

    done ;

done