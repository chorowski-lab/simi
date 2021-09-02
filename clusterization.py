import json
import os
import pathlib
from argparse import ArgumentParser
from pathlib import Path
from time import time
from urllib import parse

import numpy as np
import progressbar
import torch
from cpc.dataset import findAllSeqs
from cpc.feature_loader import buildFeature_batch

from simi.clusterization.utils_functions import loadClusterModule, loadCPCFeatureMaker, readArgs


def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('clustering_checkpoint', type=pathlib.Path,
                        help='Path to the clustering checkpoint')
    parser.add_argument('dataset', type=pathlib.Path,
                        help='Path to the dataset, which is to be quantized')
    parser.add_argument('output', type=pathlib.Path,
                        help='Output path')
    parser.add_argument('--file-ext', type=str, default='wav',
                        help='File extension of the audio files')
    parser.add_argument('--cuda', action='store_true', 
                        help='Use CUDA')
    return parser.parse_args()



def quantize_file(file_path, cpc_feature_function, clusterModule, args):
    # Get CPC features
    cFeatures = cpc_feature_function(file_path)
    if clusterModule.Ck.is_cuda:
        cFeatures = cFeatures.cuda()

    nGroups = cFeatures.size(-1)//clusterModule.Ck.size(-1) # groups information

    # Quantize the output of clustering on the CPC features
    cFeatures = cFeatures.view(1, -1, clusterModule.Ck.size(-1))

    clustered = clusterModule(cFeatures)
    if args.cuda:
        clusterModule = clusterModule.cuda()

    return clustered.detach().cpu().numpy().reshape(-1, clusterModule.k)

def main(args):
    pathClusteringCheckpoint = str(args.clustering_checkpoint) # '/pio/data/zerospeech2021/checkpoints/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt'
    pathDB = str(args.dataset) #  '/pio/data/zerospeech2021/LibriSpeech/test-clean'
    pathOutputDir = str(args.output) # '/pio/scratch/1/i290956/zs2021/clusterings/LibriSpeech/test-clean'

    seqNames, _ = findAllSeqs(pathDB, speaker_level=1, extension=args.file_ext, loadCache=True)

    if not os.path.exists(pathOutputDir):
        print("")
        print(f"Creating the output directory at {pathOutputDir}")
        Path(pathOutputDir).mkdir(parents=True, exist_ok=True)
    
    assert len(seqNames) > 0, \
        "No file to be quantized!"

    assert pathClusteringCheckpoint[-3:] == ".pt"
    if os.path.exists(pathClusteringCheckpoint[:-3] + "_args.json"):
        pathConfig = pathClusteringCheckpoint[:-3] + "_args.json"
    elif os.path.exists(os.path.join(os.path.dirname(pathClusteringCheckpoint), "checkpoint_args.json")):
        pathConfig = os.path.join(os.path.dirname(pathClusteringCheckpoint), "checkpoint_args.json")

    clustering_args = readArgs(pathConfig)
    print("")
    print(f"Clutering args:\n{json.dumps(vars(clustering_args), indent=4, sort_keys=True)}")
    print('-' * 50)

    if not os.path.isabs(clustering_args.pathCheckpoint): # Maybe it's relative path
        clustering_args.pathCheckpoint = os.path.join(os.path.dirname(os.path.abspath(pathClusteringCheckpoint)), clustering_args.pathCheckpoint)
    assert os.path.exists(clustering_args.pathCheckpoint), \
        f"CPC path at {clustering_args.pathCheckpoint} does not exist!!"
    
    # Load CluterModule
    print("")
    print(f"Loading ClusterModule at {pathClusteringCheckpoint}")
    clusterModule = loadClusterModule(pathClusteringCheckpoint)
    if args.cuda:
        clusterModule.cuda()
    print("ClusterModule loaded!")


    print("")
    print(f"Loading CPC FeatureMaker from {clustering_args.pathCheckpoint}")
    ## If we don't apply batch implementation, we can set LSTM model to keep hidden units
    ## making the quality of the quantized units better (that's why I set keep_hidden=args.nobatch)
    featureMaker = loadCPCFeatureMaker(
                        clustering_args.pathCheckpoint, 
                        gru_level=vars(clustering_args).get('level_gru', None), 
                        get_encoded=clustering_args.encoder_layer, 
                        keep_hidden=False)

    if clustering_args.dimReduction is not None:
        dimRed = loadDimReduction(clustering_args.dimReduction, clustering_args.centroidLimits)
        featureMaker = torch.nn.Sequential(featureMaker, dimRed)
    if not clustering_args.train_mode:
        featureMaker.eval()
    if args.cuda:
        featureMaker.cuda()
    def cpc_feature_function(x): 
        return buildFeature_batch(featureMaker, x,seqNorm=False, strict=True,
                                  maxSizeSeq=10240, batch_size=8)
        
    print("CPC FeatureMaker loaded!")

    # Quantization of files
    print("")
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    for index, vals in enumerate(seqNames):
        bar.update(index)

        file_path = vals[1]
        file_path = os.path.join(pathDB, file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        outputPath = os.path.join(pathOutputDir, file_name + '.npy')
        if not os.path.exists(outputPath):
            # Quantization
            f = open(outputPath, 'wb')
            f.close()
            clustered_file = quantize_file(file_path, cpc_feature_function, clusterModule, args)
            np.save(outputPath, clustered_file)

    bar.finish()
    print(f"...done {len(seqNames)} files in {time()-start_time} seconds.")


if __name__ == "__main__":
    args = parseArgs()
    main(args)
