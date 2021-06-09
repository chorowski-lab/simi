from fit import score_cpc_quantizations
import pandas

trainset = 'train-full-960'
results = pandas.DataFrame({
    'trainset': [],
    'testset': [],
    'segmentation': [],
    'vocab_size': [],
    'accuracy': [],
    'best_shift': []
})


for vocab_size in [100]: # , 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]:
    for testset in ['dev-clean']: #, 'dev-other']:
        for segmentation in ['sentencepiece']: #, 'viterbi']:
            shift_results = []
            for shift in range(-2, 3):
                segmentation_gt = f'/pio/data/zerospeech2021/librispeech_alignments/{testset}'
                segmentation_es = f'/pio/scratch/1/i290956/zs2021/simi/segmentations/{trainset}_{testset}_vs{vocab_size}/{segmentation}_segmentation'
                # quantized = f'/pio/data/zerospeech2021/quantized/LibriSpeech/{testset}/quantized_outputs.txt'

                acc = score_cpc_quantizations(segmentation_gt, segmentation_es, shift=shift)
                shift_results.append((acc, shift))
            
            accuracy = sorted(shift_results)[-1][0]
            best_shift = sorted(shift_results)[-1][1]
            print(f'{trainset} - {testset} - {vocab_size} - {segmentation}\n\tscore: {accuracy:.2f}% at shift={best_shift}')

            results = results.append({
                'trainset': trainset,
                'testset': testset,
                'segmentation': segmentation,
                'vocab_size': vocab_size,
                'accuracy': accuracy,
                'best_shift': best_shift
            }, ignore_index=True)

results.to_csv('./segmentation_results.csv', index=False)