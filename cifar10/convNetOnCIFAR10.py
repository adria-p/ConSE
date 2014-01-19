import cPickle
import time
import theano
import numpy as np
import theano.tensor as T

import climin.stops
import climin.initialize

from brummlearn.convnet import ConvNet
from brummlearn.data import one_hot

import warnings
theano.config.exception_verbosity = 'high'

"""
    Original in Maxout paper:
    batch_size: 128,
        layers: [
                 !obj:pylearn2.models.maxout.MaxoutConvC01B {
                     layer_name: 'h0',
                     pad: 4,
                     tied_b: 1,
                     W_lr_scale: .05,
                     b_lr_scale: .05,
                     num_channels: 96,
                     num_pieces: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: .9,
                     partial_sum: 33,
                 },
                 !obj:pylearn2.models.maxout.MaxoutConvC01B {
                     layer_name: 'h1',
                     pad: 3,
                     tied_b: 1,
                     W_lr_scale: .05,
                     b_lr_scale: .05,
                     num_channels: 192,
                     num_pieces: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: 1.9365,
                     partial_sum: 15,
                 },
                 !obj:pylearn2.models.maxout.MaxoutConvC01B {
                     pad: 3,
                     layer_name: 'h2',
                     tied_b: 1,
                     W_lr_scale: .05,
                     b_lr_scale: .05,
                     num_channels: 192,
                     num_pieces: 2,
                     kernel_shape: [5, 5],
                     pool_shape: [2, 2],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: 1.9365,
                 },
                 !obj:pylearn2.models.maxout.Maxout {
                    layer_name: 'h3',
                    irange: .005,
                    num_units: 500,
                    num_pieces: 5,
                    max_col_norm: 1.9
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 10,
                     irange: .005
                 }
                ],
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: &window_shape [32, 32],
            num_channels: 3,
            axes: ['c', 0, 1, 'b'],
        }
    },

"""

def convolutional_nets_on_CIFAR10():

    #### load data ####
    train_file = 'pylearn2_gcn_whitened/train.pkl'
    test_file = 'pylearn2_gcn_whitened/test.pkl'
    # Load data.

    f = open(train_file,'rb')
    train_set = cPickle.load(f)
    f = open(test_file)
    test_set = cPickle.load(f)

    X, Z = train_set.get_data()
    VX, VZ = test_set.get_data()

    Z = one_hot(Z, 10)
    VZ = one_hot(VZ, 10)

    X = X[:128*200]
    Z = Z[:128*200]
    VX = VX[:256]
    VZ = VZ[:256]


    #### initialize model ####

    max_passes = 50
    batch_size = 128
    max_iter = max_passes * X.shape[0] / batch_size
    n_report = batch_size * 50

    stop = climin.stops.any_([
        climin.stops.after_n_iterations(max_iter),
        ])

    pause = climin.stops.modulo_n_iterations(n_report)

    optimizer = 'gd', {'steprate': 0.1}
    #optimizer = dropout_optimizer_conf(steprate_0=1, n_repeats=1)
    m = ConvNet(3072, [96, 192, 192], [500], 10, ['tanh', 'tanh', 'tanh'], ['tanh'], out_transfer='softmax',
                loss='nce', image_height=32, image_width=32, n_image_channel=3, optimizer=optimizer,
                batch_size=batch_size, max_iter=max_iter, pool_shapes=[[4, 4], [4, 4], [2, 2]],
                filter_shapes=[[8, 8], [8, 8], [5, 5]], pool_strides=[[2, 2], [2, 2], [2, 2]],
                padding=[4,3,3])

    m.parameters.data[...] = np.random.normal(0, 0.1, m.parameters.data.shape)
    inits = m.init_conv_weights()
    for name, val in inits:
        m.parameters[name] = val

    n_wrong = 1 - T.eq(T.argmax(m.exprs['output'], axis=1), T.argmax(m.exprs['target'], axis=1)).mean()
    f_n_wrong = m.function(['inpt', 'target'], n_wrong)

    losses = []
    v_losses = []
    print 'max iter', max_iter

    #### train model ####

    start = time.time()
    # Set up a nice printout.
    keys = '#', 'val loss', 'seconds', 'val emp'
    max_len = max(len(i) for i in keys)
    header = '\t'.join(i for i in keys)
    print header
    print '-' * len(header)

    f_loss = m.function(['inpt', 'target'], ['loss'])

    for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause)):
        if info['n_iter'] % n_report != 0:
            continue
        passed = time.time() - start
        v_losses.append(info['val_loss'])

        #img = tile_raster_images(fe.parameters['in_to_hidden'].T, image_dims, feature_dims, (1, 1))
        #save_and_display(img, 'filters-%i.png' % i
        f_wrong_val = m.apply_minibatches_function(f_n_wrong, VX, VZ)*VX.shape[0]
        info.update({
            'time': passed,
            'val_emp': f_wrong_val
        })
        row = '%(n_iter)i\t%(val_loss)g\t%(time)g\t%(val_emp)g' % info
        print row
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        convolutional_nets_on_CIFAR10()
