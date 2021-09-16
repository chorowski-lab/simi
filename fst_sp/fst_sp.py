
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



# In classic sentencepiece this is more complex, but for our use only 1 method is necessary.
# Maybe it should not be wrapped in an object then, feel free to chenge it to a single method.
class SentencePieceTrainer(object):

    # I changed the way it works - instead of handling model saving, it simply returns it.
    # Saving will be done by a model itself, I think it's a bit simpler solution.
    #
    # The 'sentences' param should be the input/training data, in form of list(str) or list(list(int)).
    # Maybe we should also consider numpy array?
    #
    # Also, the parameters are subjects for change (specifically addition).
    @staticmethod
    def train(sentences) -> SentencePieceModel:
        raise NotImplementedError()


