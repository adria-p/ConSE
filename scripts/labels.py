__author__ = 'kosklain'

import os
import cPickle
import numpy as np

def unpickle_labels(file_name):
    fo = open(file_name, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary["labels"]


num_batches = 5
files_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cifar10", "cifar-10-batches-py")
files_name = "data_batch_{}"
test_name = "test_batch"
files_save = "data_labels.npy"
test_save = "test_labels.npy"

labels_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cifar10")

labels = []

for batch in range(1,num_batches+1):
    file_name = os.path.join(files_dir, files_name.format(batch))
    labels += unpickle_labels(file_name)

test_batch_name = os.path.join(files_dir, test_name)
test_labels = unpickle_labels(test_batch_name)

np_labels = np.array(labels)
np_test_labels = np.array(test_labels)
np.save(os.path.join(labels_save_dir, files_save),np_labels)
np.save(os.path.join(labels_save_dir, test_save), np_test_labels)