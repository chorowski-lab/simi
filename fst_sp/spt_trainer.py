from typing import Optional
from collections import Counter
from spt_model import SentencePieceModel
from utils import Sentence,SentencePiece,PieceCounts,ViterbiPath,EStepRet
from esa import ESA
import numpy as np


class SentencePieceTrainerParameters(object):

    defaults = { 
        "vocab_size": 90,
        "max_piece_length": 1000,
        'num_sub_iterations': 2, #EM subiterations
        'seed_vocab_size': 10000000,
        'verbose': False, #more information printed
        }

    def __init__(self,parameters=None):
        if parameters:
            raise NotImplementedError()
        
        self.vocab_size         = SentencePieceTrainerParameters.defaults["vocab_size"]
        self.max_piece_length   = SentencePieceTrainerParameters.defaults["max_piece_length"]
        self.num_sub_iterations = SentencePieceTrainerParameters.defaults["num_sub_iterations"]
        self.verbose            = SentencePieceTrainerParameters.defaults["verbose"]
        self.seed_vocab_size    = SentencePieceTrainerParameters.defaults["seed_vocab_size"]



class SentencePieceTrainer(object):

    def __init__(self, parameters:Optional[SentencePieceTrainerParameters] = None):

        if parameters:
            self.parameters = parameters
        else:
            self.parameters = SentencePieceTrainerParameters() # get defaults

    @staticmethod
    def get_seed_pieces(sentences, seed_vocab_size, max_piece_length, debug=False):
        def to_log_prob(pieces):
            Z = np.log(sum(score for p, score in pieces))
            pieces = [(p, np.log(score) - Z) for p, score in pieces]
            return pieces
        print("Extracting frequent sub strings...")

        # Makes an enhanced suffix array to extract all sub strings occurring
        # more than 2 times in the sentence.
        
        delimiter=u'\u25C6'

        esa = ESA()
        esa.fit([], delimiter=delimiter, max_piece_len=max_piece_length, debug = debug)

        seed_sentp = sorted(esa.pieces(), key=lambda p_score: -p_score[1]) 

        #TODO: bug(?) - single letters (all?) are added by esa, temporary workaround:
        seed_sentp = [x for x in seed_sentp if len(x[0])>1]
            
        # Prune
        seed_sentp = seed_sentp[:seed_vocab_size]

        # all_chars must be included in the seed sentencepieces.
        all_chars = Counter()
        for s in sentences:
            all_chars.update(s)
        del all_chars[delimiter]

        for c, cnt in all_chars.items():
            seed_sentp.append((c, cnt))  # 0.5)) # XXX XXX XXX


        seed_sentp = to_log_prob(seed_sentp)
        seed_sentp = sorted(seed_sentp, key=lambda p_score: -p_score[1])
        seed_sentp.insert(0,("<unk>",0))
        seed_sentp.insert(1,("<s>",0))
        seed_sentp.insert(2,("</s>",0)) 

        #print(" ".join(s for s,c in seed_sentp[:50]))

        print(f"Initialized {len(seed_sentp)} seed sentencepieces")
        return [SentencePiece(ind,symb,freq) for ind,(symb,freq) in enumerate(seed_sentp)]


    # I changed the way it works - instead of handling model saving, it simply returns it.
    # Saving will be done by a model itself, I think it's a bit simpler solution.
    #
    # The 'sentences' param should be the input/training data, in form of list(str) or list(list(int)).
    # Maybe we should also consider numpy array?
    #
    # Also, the parameters are subjects for change (specifically addition).
    def train(self,sentences) -> SentencePieceModel:

        types = [type(s) for s in sentences]

        if all(t == str for t in types):
            sentences = [Sentence( (" " + s.strip()).replace(" ","▁"), 1) for s in sentences] #_This_format_of_sequence
        elif all(t == Sentence for t in types):
            sentences = [Sentence( (" " + s.strip()).replace(" ","▁"), c) for ( s,c ) in sentences] #_This_format_of_sequence
        else:
            raise NotImplementedError()



        seed_sp = SentencePieceTrainer.get_seed_pieces(sentences, #In esa
                        seed_vocab_size = self.parameters.seed_vocab_size, 
                        max_piece_length = self.parameters.max_piece_length,
                        debug = self.parameters.verbose)


        #Sentencepiece training
        pieces = seed_sp 
        T=fst.SentencePieceTrainer(pieces) #in kaldi_fst_sp
        sentences = [fst.Sentence(text, 1) for text in sentences]

        vocab_size = self.parameters.vocab_size
        prune_fact = self.parameters.shrinking_factor
        num_subiter = self.parameters.num_sub_iterations

        while True:
            for sub_iter in range(num_subiter):  
                e_ret = T.EStep(pieces, sentences)
                pieces = T.MStep(pieces, e_ret.counts)
                print(f"EM sub_iter={sub_iter} size={len(pieces)} tot_piece_prob={np.exp(logsumexp([piece.log_freq for piece in pieces]))} "
                    f"obj={e_ret.objective} num_tokens={e_ret.n_tokens} num_tokens/piece={e_ret.n_tokens / len(pieces)}" )
            
            if len(pieces) <= vocab_size: 
                break

            pieces = T.prune_pieces(pieces, sentences, vocab_size, prune_fact)

            if len(pieces) <= vocab_size: 
                break

        pieces = sorted(pieces,key= lambda x: -x.log_freq)        


