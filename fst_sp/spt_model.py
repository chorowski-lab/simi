#from spt_trainer import SentencePieceTrainer
import fst_tools

class SentencePieceModel(object):

    def __init__(self, prefix=None, model=None, vocab=None):
        # It should load itself either from prefix or initialize from given data (eg. model+vocab)
        if vocab and model:
            self.vocab=vocab
            self.sp_to_char = model
        if prefix:
            raise NotImplementedError()

    def encode(self, sentences):
        # It should segment the sentences with some kind of viterbi. Sentences should be the same format as the training data.
        vit_paths = [fst_tools.viterbi(s, self.sp_to_char, nshortest=1)[0] for s in sentences]
        wordseq = [[self.vocab[i][1] for i in v[0]] for v in vit_paths]
        return wordseq

    def save(self, prefix):
        # It should save the model (data+vocab) at the specified location/path.
        raise NotImplementedError()
