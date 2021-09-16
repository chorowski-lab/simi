class SentencePieceModel(object):

    def __init__(self, prefix=None, model=None, vocab=None):
        # It should load itself either from prefix or initialize from given data (eg. model+vocab)
        raise NotImplementedError()

    def encode(self, sentences):
        # It should segment the sentences with some kind of viterbi. Sentences should be the same format as the training data.
        raise NotImplementedError()

    def save(self, prefix):
        # It should save the model (data+vocab) at the specified location/path.
        raise NotImplementedError()
