from simi.fst_sp.kaldi_fst_sp import Sentence
from spt_model import *
from utils import *


class SentencePieceTrainerParameters(object):

    defaults = { 
        "vocab_size": 90,
        "max_piece_length": 1000,
        'num_sub_iterations': 2, #EM subiterations
        'verbose': True, #more information printed
        }

    def __init__(self,parameters=None):
        if parameters:
            raise NotImplementedError()
        
        self.vocab_size         = SentencePieceTrainerParameters.defaults["vocab_size"]
        self.max_piece_length   = SentencePieceTrainerParameters.defaults["max_piece_length"]
        self.num_sub_iterations = SentencePieceTrainerParameters.defaults["num_sub_iterations"]
        self.verbose            = SentencePieceTrainerParameters.defaults["verbose"]



class SentencePieceTrainer(object):

    def __init__(self, parameters:SentencePieceTrainerParameters = None):

        if self.parameters:
            self.parameters = parameters
        else:
            self.parameters = SentencePieceTrainerParameters() # get defaults


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
            sentences = [Sentence( (" " + s.strip()).replace(" ","▁"), c) for Sentence( s,c ) in sentences] #_This_format_of_sequence
        else:
            raise NotImplementedError()



        seed_sp = make_seed_sentence_pieces([s for Sentence(s,c) in sentences], #In esa
                        seed_vocab_size = self.parameters.seed_sentencepiece_size, 
                        max_piece_length = self.parameters.max_piece_length,
                        debug = self.parameters.verbose)

        seed_sp.insert(0,("<unk>",0))
        seed_sp.insert(1,("<s>",0))
        seed_sp.insert(2,("</s>",0)) 


        #Sentencepiece training
        pieces = [fst.SentencePiece(ind,symb,log_freq) for ind,(symb,log_freq) in enumerate(seed_sp)] #in kaldi_fst_sp
        T=fst.SentencePieceTrainer(pieces)
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


