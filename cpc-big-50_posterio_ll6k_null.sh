
: ${ZS:="/pio/data/zerospeech_nips2021"}
: ${POSTERIO_PATH:="/pio/gluster/santiago/posteriorograms/ll6k_50_nullspace_2"}
CPC_CHECKPOINT="/pio/scratch/1/i323106/wav2vec/runs/cpc/cpc_big_ll6k/kmeans_null_2/50clusters.pt"
: ${AUDIO_DATASET_PATH:="/pio/data/libri-light-6k/medium_cut_by_vad_60"}

echo $POSTERIO_PATH
mkdir -p $POSTERIO_PATH

export CUDA_VISIBLE_DEVICES=1

PYTHONPATH='.':$PYTHONPATH python -u make_posteriograms.py \
    --cuda \
    --file-ext flac \
    $CPC_CHECKPOINT \
    $AUDIO_DATASET_PATH \
    $POSTERIO_PATH \
    $@

chmod 777 -R $POSTERIO_PATH

exit 0
