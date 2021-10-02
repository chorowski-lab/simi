#!/bin/bash

python experiment.py \
    ./models/segmentations/train-clean-100_train-clean-100_vs100000/sentencepiece_segmentation/reduced_vs50000/word_map.txt \
    ./models/sentencepieces/train-clean-100_vs100000 \
    /pio/data/zerospeech2021/quantized/LibriSpeech/train-clean-100/quantized_outputs.txt \
    ./model_100