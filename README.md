# SIMI experiments

Repository containing current experiments related to the simi (semantic) task of the ZeroSpeech 2021 Challenge.

## Scoring

**Example command**: `./scoring/run.sh`

We compute the PER (Phone Error Rate) of a clusterization by greedy-mapping every sentence piece to a most frequent ground-truth phone. Then we do a greedy alignment, and compute the mismatch error rate.

Parameters:
- `gt`: path to the ground-truth segmentation, eg.: `/pio/data/zerospeech2021/librispeech_alignments/<librispeech_subset>`
- `quantized`: path to the segmentation you want to rate
- `frame_shift`: offset in number of frames, in most cases it should be 0

## Clustering

**Example command**: `./simi/clustering/run.sh`

This is a script for running the CPC_kmeans checkpoint, but without the latest argmax (we leave each frame as distribution on centroids). This is required for the viterbi segmentation.

To get the description of parameters run: `python clustering.py --help`.
## Segmentation

**Example command**: `./simi/segmentation/run.sh`

We run sentencepiece on given `trainset`, then using learnt language model we try to predict the best segmentation. There are two ways of doing so:
1. use sentencepiece's default segmentation (by default), but it's bad because our language is not very "exact" - some pseudophones (output of the CPC+clustering) may be mismatched
2. use viterbi segmentation (flag `--viterbi`). It takes errors into account, and it usually gives lower PER. 

To get the description of parameters run: `python segmentation.py --help`.

## Grouping

**Example command**: `./simi/grouping/run.sh`

The idea is based on the fact that segmentation using sentencepiece requires large vocab to work, but it results in multiple different sentence pieces mapping to the same phoneme. In order to reduce number of sentence pieces, we run word2vec on them and then k-means, grouping and merging multiple sentencepieces together. It never increases PER, but (greatly) reduces vocab size.

To get the description of parameters run: `python grouping.py --help`.

## Statistics

In folder `simi/statistics` there are python scripts for computing various statistics.

### Pieces

Command: `python simi/statistics/pieces.py <segmentation_path>`

It prints the number of pieces and max/min/mean piece counts.
