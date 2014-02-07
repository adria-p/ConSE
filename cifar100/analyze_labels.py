__author__ = 'adria'
import os
from gensim.models.word2vec import Word2Vec

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

labels = unpickle('meta')['fine_label_names']
cifar100 = os.path.dirname(os.path.realpath(__file__))
text = os.path.join(os.path.dirname(cifar100), 'text')
model_path = os.path.join(text, 'text9model.model')
model = Word2Vec.load(model_path)
vocabulary = [key for key in model.__dict__['vocab']]
not_found_labels = []
for label in labels:
    found = False
    for vocab in vocabulary:
        if label == vocab:
            found = True
            break
    if not found:
        not_found_labels.append(label)
print not_found_labels