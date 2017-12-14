import os

from gensim.models import utils
from gensim.models import Doc2Vec

from llc import LabeledLineSentence

train_epoch = 5
model_path = './model.d2v'

# if model exists, preload
# TODO: console arguments to not load
if (os.path.isfile(model_path)):
    print("Model found! Importing...")
    model = Doc2Vec.load(model_path)

# data setup
sources = {'./bbc/train_bs.txt':'BUSINESS', './bbc/train_en.txt':'ENTERTAINMENT',
           './bbc/train_pol.txt':'POLITICS', './bbc/train_sp.txt':'SPORTS',
           './bbc/train_te.txt':'TECHNOLOGY'}

sentences = LabeledLineSentence(sources)

# build vocab table
print("Building vocab...")
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
model.build_vocab(sentences.to_array())

# train model
print("Training...")
for epoch in range(train_epoch):
    model.train(sentences.sentences_perm(),
    total_examples=model.corpus_count, epochs=train_epoch)

# save model
print("Done! Saving trained model...")
model.save(model_path)
