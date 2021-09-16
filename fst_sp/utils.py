from collections import defaultdict, namedtuple

Sentence = namedtuple('Sentence', ['text', 'count'])
SentencePiece = namedtuple('SentencePiece', ['index', 'symbol', 'log_freq'])
PieceCounts = namedtuple('PieceCounts', ['Z', 'counts'])
ViterbiPath = namedtuple('ViterbiPath', ['path', 'prob', 'log_prob'])
EStepRet = namedtuple('EStepRet', ['objective', 'n_tokens', 'counts'])