__author__ = 'kosklain'

# -*- coding: utf-8 -*-

import cPickle
import gzip
import time

import numpy as np
import theano.tensor as T

import climin.stops
import climin.initialize

from brummlearn.convnet import ConvNet
from brummlearn.data import one_hot

# <markdowncell>

# Prepare Data
# ============

# <codecell>
import pylearn2.model

datafile = 'mnist.pkl.gz'
# Load data.

with gzip.open(datafile,'rb') as f:
    train_set, val_set, test_set = cPickle.load(f)

X, Z = train_set
VX, VZ = val_set
TX, TZ = test_set

Z = one_hot(Z, 10)
VZ = one_hot(VZ, 10)
TZ = one_hot(TZ, 10)

image_dims = 28, 28

# <markdowncell>

# Define Model
# ============

# <codecell>

max_passes = 100
batch_size = 1000
max_iter = max_passes * X.shape[0] / batch_size
n_report = X.shape[0] / batch_size

stop = climin.stops.any_([
    climin.stops.after_n_iterations(max_iter),
    ])

pause = climin.stops.modulo_n_iterations(n_report)

optimizer = 'rmsprop', {'steprate': 0.001, 'momentum': 0.9, 'decay': 0.9, 'step_adapt': 0.01}
#optimizer = dropout_optimizer_conf(steprate_0=1, n_repeats=1)
m = ConvNet(784, [20, 50], [500], 10, ['tanh', 'tanh', 'tanh'], ['sigmoid'], out_transfer='softmax',
            loss='nce', image_height=28, image_width=28, n_image_channel=1, optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)

m.parameters.data[...] = np.random.normal(0, 1, m.parameters.data.shape)

weight_decay = ((m.parameters.in_to_hidden**2).sum()
                #+ (m.parameters.hidden_conv_to_hidden_conv_0**2).sum()
                + (m.parameters.hidden_conv_to_hidden_full**2).sum()
                #+ (m.parameters.hidden_full_to_hidden_full_0**2).sum()
                + (m.parameters.hidden_to_out**2).sum())
weight_decay /= m.exprs['inpt'].shape[0]
m.exprs['true_loss'] = m.exprs['loss']
c_wd = 0.00001
m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

n_wrong = 1 - T.eq(T.argmax(m.exprs['output'], axis=1), T.argmax(m.exprs['target'], axis=1)).mean()
f_n_wrong = m.function(['inpt', 'target'], n_wrong)

losses = []
v_losses = []
print 'max iter', max_iter

# <markdowncell>

# Learn
# =====
#
# First train an report validation error to manually check for the training error at which validation error is minimal.

# <codecell>

start = time.time()
# Set up a nice printout.
keys = '#', 'loss', 'val loss', 'seconds', 'wd', 'train emp', 'val emp'
max_len = max(len(i) for i in keys)
header = '\t'.join(i for i in keys)
print header
print '-' * len(header)

f_loss = m.function(['inpt', 'target'], ['true_loss', 'loss'])

for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause)):
    if info['n_iter'] % n_report != 0:
        continue
    passed = time.time() - start
    losses.append(info['loss'])
    v_losses.append(info['val_loss'])

    #img = tile_raster_images(fe.parameters['in_to_hidden'].T, image_dims, feature_dims, (1, 1))
    #save_and_display(img, 'filters-%i.png' % i)
    f_wrong = m.apply_minibatches_function(f_n_wrong, X, Z)*X.shape[0]
    f_wrong_val = m.apply_minibatches_function(f_n_wrong, VX, VZ)*VX.shape[0]
    info.update({
        'time': passed,
        'train_emp': f_wrong,
        'val_emp': f_wrong_val
    })
    row = '%(n_iter)i\t%(loss)g\t%(val_loss)g\t%(time)g\t%(train_emp)g\t%(val_emp)g' % info
    print row


# <markdowncell>

# Final Training
# --------------
# Train on train+validation until the training error passes a threshold.

# <codecell>

m.parameters.data[...] = np.random.normal(0, 1e-3, m.parameters.data.shape)

start = time.time()
# Set up a nice printout.
keys = '#', 'loss', 'val loss', 'seconds'#, 'step_length'
max_len = max(len(i) for i in keys)
header = '   '.join(i.ljust(max_len) for i in keys)
print header
print '-' * len(header)

RVX = np.concatenate([X, VX], axis=0)
RVZ = np.concatenate([Z, VZ], axis=0)

for i, info in enumerate(m.powerfit((RVX, RVZ), (VX, VZ), stop, pause)):
    passed = time.time() - start
    losses.append(info['loss'])
    v_losses.append(info['val_loss'])

    row = '%i' % info['n_iter'], '%.6f' % info['loss'], '%.6f' % info['val_loss'], '%.3f' % passed#, '%.6f' % info['step_length']
    print '   '.join(i.ljust(max_len) for i in row)

    if losses[-1] <= 0.023:
        break

# <codecell>

f_predict = m.function(['inpt'], T.argmax(m.exprs['output_in'], axis=1))

TY = f_predict(TX)

print '#wrong', (TY != TZ.argmax(axis=1)).sum()

# <codecell>
