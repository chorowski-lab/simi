import sentencepiece as spm
from simi.fst_sp.kaldi_fst_sp import *
import io

model = io.BytesIO()

spm.SentencePieceTrainer.train(
    input ="simi/fst_sp/botchan_small.txt",
    model_writer=model,
    vocab_size=100, 
    shrinking_factor=0.75,
    #use_all_vocab=True,
    #shuffle_input_sentence=0,
    num_sub_iterations=2,
    max_sentencepiece_length = 10,
)

sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
FINAL_PIECES = list(extract_pieces(sp))
for i,s,l in FINAL_PIECES:
    print(i,s,l)
