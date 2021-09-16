from spt_trainer import SentencePieceTrainer,SentencePieceModel


# EXAMPLE 1

# list of strings
string_data = list(line.strip() for line in open('./botchan.txt', 'r', encoding='utf8'))
# train model
trainer = SentencePieceTrainer()
model = trainer.train(string_data)
# example sentence (string)
sentence = 'I saw a girl with a telescope.'
# segmenting the sentence
encoding = model.encode(sentence)

# assert encoding == ['I', ' saw', ' a', ' girl', ' with', ' a', ' ', 'te', 'le', 's', 'c', 'o', 'pe', '.']


# EXAMPLE 2

# list of lists of ints
int_data = list(map(lambda s: list(map(ord, s)), string_data))
# train model
trainer = SentencePieceTrainer()
model = trainer.train(int_data)
# example sentence (list of ints)
sentence = list(map(ord, 'I saw a girl with a telescope.'))
# segmenting the sentence
encoding = model.encode(sentence)

# assert encoding == [[73], [32, 115, 97, 119], [32, 97], [32, 103, 105, 114, 108]...


# EXAMPLE 3
path = '/some/path/on/disk'
model.save(path)
model2 = SentencePieceModel(prefix=path)
assert model2.encode(sentence) == encoding
