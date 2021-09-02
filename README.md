# SIMI experiments

Repository containing current experiments related to the simi (semantic) task of the ZeroSpeech 2021 Challenge.

## Scoring and statistics

### Simi

We can score phone-level segmentation by running it through a simple SIMI solution and then rate it with official [`zerospeech2021-evaluate`](https://github.com/bootphon/zerospeech2021) tool. In order to do so we will need phone-segmented dataset and `semantic/dev/**` subsets. An example on how to prepare such segmentations is in script [simi/task/prepare_segmentation.sh](simi/task/prepare_segmentation.sh).

To score segmentations use the file [simi/task/run.sh](simi/task/run.sh). You will need to change these parameters:
- `VOCAB_SIZE` - this is the size of the sentencepiece's vocabulary, should be roughly 50000
- `TRAINSET_PATH` - path to the phonemized data set
- `TESTSET_(LIBRISPEECH|SYNTHETIC)_PATH` - path to the phonemized test sets: `semantic/dev/librispeech` and `semantic/dev/synthetic`
- `OUTPUT_PATH` - directory where two folders (`librispeech` and `synthetic`) consisting of semantic vectors will be created, in format accepted by [`zerospeech2021-evaluate`](https://github.com/bootphon/zerospeech2021).

### PER

**Example script**: [scoring/run.sh](scoring/run.sh)

We compute the PER (Phone Error Rate) of a clusterization by greedy-mapping every sentence piece to a most frequent ground-truth phone. Then we do a greedy alignment, and compute the mismatch error rate.

Parameters:
- `gt`: path to the ground-truth segmentation, eg.: `/pio/data/zerospeech2021/librispeech_alignments/<librispeech_subset>`
- `quantized`: path to the segmentation you want to rate
- `frame_shift`: offset in number of frames, in most cases it should be 0

### Number of pieces

Command: `python simi/statistics/pieces.py <segmentation_path>`

It prints the number of pieces and max/min/mean piece counts.

## Clustering

**Example script**: [simi/clusterization/run.sh](simi/clusterization/run.sh)

This is a script for running the CPC_kmeans checkpoint, but without the latest argmax (we leave each frame as distribution on centroids). This is required for the viterbi segmentation.

To get the description of parameters run: `python clusterization.py --help`.
## Segmentation

**Example script**: [simi/segmentation/run.sh](simi/segmentation/run.sh)

We run sentencepiece on given `trainset`, then using learnt language model we try to predict the best segmentation. There are two ways of doing so:
1. use sentencepiece's default segmentation (by default), but it's bad because our language is not very "exact" - some pseudophones (output of the CPC+clustering) may be mismatched
2. use viterbi segmentation (flag `--viterbi`). It takes errors into account, and it usually gives lower PER. 

To get the description of parameters run: `python segmentation.py --help`.

You can also train sentencepiece and evaluate segmentation separately, using these two scripts: [train_sentencepiece.py](train_sentencepiece.py) and [eval_segmentation.py](eval_segmentation.py). Run `python <SCRIPT_NAME> --help` for more info.

## Grouping

**Example script**: [simi/grouping/run.sh](simi/grouping/run.sh)

The idea is based on the fact that segmentation using sentencepiece requires large vocab to work, but it results in multiple different sentence pieces mapping to the same phoneme. In order to reduce number of sentence pieces, we run word2vec on them and then k-means, grouping and merging multiple sentencepieces together. It never increases PER, but (greatly) reduces vocab size.

To get the description of parameters run: `python grouping.py --help`.
