__author__ = 'adria'
import numpy as np
from collections import defaultdict
from cifar100.convNetOnCIFAR100 import Cifar100Worker
from gensim.models.word2vec import Word2Vec
import os
from breze.learn.cnn import Cnn

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def get_top_n(vector, n, gt):
    dists = np.dot(text_model.syn0norm, vector)
    best = np.argsort(dists)[::-1]
    for i, w in enumerate(best):
        if text_model.index2word[w] == gt:
            return i
    raise RuntimeError('It should be there!')

def get_word(idx):
    word = labels[idx]
    if word in word_transf:
        word = word_transf[word]
    return word

def load_model(filename, n_outputs):
    m = Cnn(3072, [64, 64, 64], [256], n_outputs,
            ['rectifier', 'rectifier', 'rectifier'], ['rectifier'],
            out_transfer='softmax', loss='nce', image_height=32,
            image_width=32, n_image_channel=3, optimizer=None,
            batch_size=128, max_iter=12,
            pool_shapes=[[3, 3], [3, 3], [3, 3]],
            filter_shapes=[[5, 5], [5, 5], [5, 5]],
            pool_strides=[[2, 2], [2, 2], [2, 2]], padding=[2,2,2],
            lrnorm=[True, True, False],
            init_weights_stdev=[0.01, 0.1, 0.1, 0.1, 0.1])
    m.parameters.data[:] = np.load(filename)
    return m

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


labels = unpickle('cifar100/cifar-100-python/meta')['fine_label_names']
labels_indices = defaultdict()
for i, label in enumerate(labels):
    labels_indices[label] = i


folder = os.path.dirname(os.path.realpath(__file__))
text = os.path.join(folder, 'text')
text_model_path = os.path.join(text, 'text9model.model')
text_model = Word2Vec.load(text_model_path)
text_model.init_sims()

word_to_directly_exclude = labels_indices['aquarium_fish']
assert word_to_directly_exclude is not None
repetitions = 10
random_words = 10
top_words = 50
seed = 10
train_file = 'cifar100/pylearn2_gcn_whitened/train.pkl'
test_file = 'cifar100/pylearn2_gcn_whitened/test.pkl'

rng = np.random.RandomState(seed)
res = 0.0
pos = 0.0
for i in xrange(repetitions):
    # Contains indices to remove once the word "aquarium fish" is already
    # removed
    indices_to_remove = range(len(labels)-1)
    rng.shuffle(indices_to_remove)
    indices_to_remove = indices_to_remove[:random_words]
    print indices_to_remove
    print "Words removed"
    to_print = []
    for j in indices_to_remove:
        if j >= word_to_directly_exclude:
            to_print.append(get_word(j+1))
        else:
            to_print.append(get_word(j))
    print to_print
    print len(to_print)
    # Contains all indices not removed, except "aquarium fish"
    indices_not_removed = []
    for j in xrange(len(labels)-1):
        if j not in indices_to_remove:
            indices_not_removed.append(j)
    indices_not_removed = np.array(indices_not_removed)
    indices_not_removed[indices_not_removed >= word_to_directly_exclude] += 1
    #Build the transformation matrix for the predictions
    print "Words not removed"
    print [get_word(idx) for idx in indices_not_removed]
    print len(indices_not_removed)
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
    print "Number of instances excluded", len(X)
    X = X[:len(X)/c.batch_size*c.batch_size]
    Z = Z[:len(Z)/c.batch_size*c.batch_size]

    # Get the original ground truth
    Z[Z >= word_to_directly_exclude] += 1
    
    if os.path.isfile("model-%i.npy" % i):
        m = load_model("model-%i.npy" % i, 99-random_words)
    else:
        m = c.run()
    predictions = m.predict(X)
    res = 0.0
    positions = 0.0
    print "Predicting..."
    for j, (pred, gt) in enumerate(zip(predictions, Z)):
        if (j % 1000) == 0:
            print "%i..." % j
        prediction_word_vector = np.dot(pred, transformation_matrix)
        prediction_word_vector /= np.sum(pred)
        prediction_word_vector /= np.linalg.norm(prediction_word_vector)
        position = get_top_n(prediction_word_vector, top_words, get_word(gt))
        #print get_word(gt), top_n_words
        if position < top_words:
            res += 1.0
        positions += position

    average_res = res/len(predictions)
    average_pos = positions/len(predictions)
    pos += average_pos
    res += average_res
    print "Result for repetition %i with top %i words: %f accuracy, " \
          "%f average position" % (i, top_words, average_res, average_pos)

print "Overal results: %f accuracy, %f average position" % (res/repetitions,
                                                            pos/repetitions)