__author__ = 'adria'
import numpy as np
from collections import defaultdict
from cifar100.convNetOnCIFAR100 import Cifar100Worker
from gensim.models.word2vec import Word2Vec
import os

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def get_top_n(vector, n):
    dists = np.dot(text_model.syn0norm, vector)
    best = np.argsort(dists)[::-1][:n]
    return [text_model.index2word[sim] for sim in best]

def get_word(idx):
    word = labels[idx]
    if word in word_transf:
        word = word_transf[word]
    return word

def compare_words(word_list, target):
    if target in word_list:
        print "Found!"
        return True
    return False

#Set the transformation and inverse trasnformations
word_transf = {}
word_transf['palm_tree'] = 'palm'
word_transf['pickup_truck'] = 'pickup'
word_transf['lawn_mower'] = 'lawnmower'
word_transf['maple_tree'] = 'maple'
word_transf['oak_tree'] = 'oaktree'
word_transf['pine_tree'] = 'pinetree'
word_transf['sweet_pepper'] = 'pepper'
word_transf['willow_tree'] = 'willow'
for key in word_transf:
    word_transf[word_transf[key]] = key



labels = unpickle('meta')['fine_label_names']
labels_indices = defaultdict()
for i, label in enumerate(labels):
    labels_indices[label] = i


folder = os.path.dirname(os.path.realpath(__file__))
text = os.path.join(folder, 'text')
text_model_path = os.path.join(text, 'text9model.model')
text_model = Word2Vec.load(text_model_path)
text_model.init_sims()

word_to_directly_exclude = labels_indices['aquarium fish']
assert word_to_directly_exclude is not None
repetitions = 10
random_words = 10
top_words = 10
seed = 10
train_file = 'cifar100/pylearn2_gcn_whitened/train.pkl'
test_file = 'cifar100/pylearn2_gcn_whitened/test.pkl'

rng = np.random.RandomState(seed)
for i in range(repetitions):
    # Contains indices to remove once the word "aquarium fish" is already
    # removed
    indices_to_remove = rng.randint(len(labels)-1, size=random_words)
    # Contains all indices not removed, except "aquarium fish"
    indices_not_removed = []
    for i in range(len(labels)-1):
        if i not in indices_to_remove:
            indices_not_removed.append(i)
    indices_not_removed = np.array(indices_not_removed)
    indices_not_removed[indices_not_removed >= word_to_directly_exclude] += 1
    #Build the transformation matrix for the predictions
    transformation_matrix = []
    for idx in indices_not_removed:
        word_vector = text_model[get_word(idx)]
        word_vector /= np.linalg.norm(word_vector)
        transformation_matrix.append(word_vector)
    transformation_matrix = np.array(transformation_matrix)

    c = Cifar100Worker(train_file=train_file, test_file=test_file,
                       remove_indices=[word_to_directly_exclude],
                       model_name='model-%i' % i)
    #Get the excluded labels
    X, Z = c.set_data_slice(indices_to_remove)
    X = X[:len(X)/c.batch_size*c.batch_size]
    Z = Z[:len(Z)/c.batch_size*c.batch_size]

    # Get the original ground truth
    Z[Z >= word_to_directly_exclude] += 1

    m = c.run()
    predictions = m.predict(X)
    res = 0.0
    for pred, gt in zip(predictions, Z):
        prediction_word_vector = np.dot(pred, transformation_matrix)
        prediction_word_vector /= np.sum(pred)
        prediction_word_vector /= np.linalg.norm(prediction_word_vector)
        top_n_words = get_top_n(prediction_word_vector, 10)
        print top_n_words
        if compare_words(top_n_words, get_word(gt)):
            res += 1.0
    print "Result for repetition %i: %f" % (i, res/len(predictions))



