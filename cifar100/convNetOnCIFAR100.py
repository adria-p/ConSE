import cPickle
import time
import theano
import numpy as np
import theano.tensor as T

import climin.stops
import climin.initialize

from breze.learn.cnn import Cnn
from breze.learn.data import one_hot

theano.config.exception_verbosity = 'high'


class Cifar100Worker:
    def __init__(self, train_file='pylearn2_gcn_whitened/train.pkl',
                 test_file='pylearn2_gcn_whitened/test.pkl', batch_size=128,
                 max_passes=40, remove_indices=[], model_name='model.model'):
        f = open(train_file, 'rb')
        train_set = cPickle.load(f)
        f = open(test_file)
        test_set = cPickle.load(f)
        X, Z = train_set.get_data()
        VX, VZ = test_set.get_data()

        self.data = [X, Z, VX, VZ]
        self.set_data_slice(remove_indices)
        self.batch_size = batch_size
        self.max_passes = max_passes
        self.model_name = model_name

    def set_data_slice(self, indices_to_remove):
        to_return_X = []
        to_return_Z = []
        to_delete = []
        for i, (x, z) in enumerate(zip(self.data[0], self.data[1])):
            if z in indices_to_remove:
                to_return_X.append(x)
                to_return_Z.append(z)
                to_delete.append(i)
        self.data[0] = np.delete(self.data[0], to_delete)
        self.data[1] = np.delete(self.data[1], to_delete)
        for i, (x, z) in enumerate(zip(self.data[2], self.data[3])):
            if z in indices_to_remove:
                to_return_X.append(x)
                to_return_Z.append(z)
                to_delete.append(i)
        self.data[2] = np.delete(self.data[2], to_delete)
        self.data[3] = np.delete(self.data[3], to_delete)
        for i in np.sort(indices_to_remove)[::-1]:
            self.data[1][self.data[1] >= i] -= 1
            self.data[3][self.data[3] >= i] -= 1

        return np.array(to_return_X), np.array(to_return_Z)

    def prepare_data(self, data_to_prepare):
        num_batches = len(data_to_prepare)/self.batch_size
        data_to_prepare[:] = data_to_prepare[:num_batches*self.batch_size]
        data_to_prepare[:] = np.array(data_to_prepare, dtype=np.float32)

    def run(self):
        self.data[1] = one_hot(self.data[1], np.max(self.data[1])+1)
        self.data[3] = one_hot(self.data[3], np.max(self.data[3])+1)
        for element in self.data:
            self.prepare_data(element)
        X, Z, VX, VZ = self.data
        max_iter = self.max_passes * X.shape[0] / self.batch_size
        n_report = X.shape[0] / (5*self.batch_size)
        stop = climin.stops.any_([
            climin.stops.after_n_iterations(max_iter),
            ])

        pause = climin.stops.modulo_n_iterations(n_report)
        optimizer = 'gd', {'steprate': 0.01, 'momentum': 0.9}
        m = Cnn(3072, [64, 64, 64], [256], 100, #256 best
                ['rectifier', 'rectifier', 'rectifier'], ['rectifier'],
                out_transfer='softmax', loss='nce', image_height=32,
                image_width=32, n_image_channel=3, optimizer=optimizer,
                batch_size=self.batch_size, max_iter=max_iter,
                pool_shapes=[[3, 3], [3, 3], [3, 3]],
                filter_shapes=[[5, 5], [5, 5], [5, 5]],
                pool_strides=[[2, 2], [2, 2], [2, 2]], padding=[2,2,2],
                lrnorm=[True, True, False],
                init_weights_stdev=[0.01, 0.1, 0.1, 0.1, 0.1])

        weight_decay = 0.4*((m.parameters.in_to_hidden**2).sum()) + \
                       0.4*((m.parameters.hidden_conv_to_hidden_conv_0**2)
                            .sum()) + \
                       0.4*((m.parameters.hidden_conv_to_hidden_conv_1**2)
                            .sum()) \
                       + 2*(m.parameters.hidden_conv_to_hidden_full**2).sum()
        weight_decay /= m.exprs['inpt'].shape[0]
        m.exprs['true_loss'] = m.exprs['loss']
        m.exprs['loss'] += weight_decay

        n_wrong = 1 - T.eq(T.argmax(m.exprs['output'], axis=1),
                           T.argmax(m.exprs['target'], axis=1)).mean()
        f_n_wrong = m.function(['inpt', 'target'], n_wrong)

        v_losses = []
        print 'max iter', max_iter

        start = time.time()
        keys = '#', 'val loss', 'seconds', 'train emp', 'val emp'
        header = '\t'.join(i for i in keys)
        print header
        print '-' * len(header)
        info = None
        for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause,
                                            eval_train_loss=False)):
            if info['n_iter'] % n_report != 0:
                continue
            passed = time.time() - start
            v_losses.append(info['val_loss'])
            f_wrong_val = m.apply_minibatches_function(f_n_wrong, VX, VZ)
            f_wrong_val = f_wrong_val*VX.shape[0]
            f_wrong_train = m.apply_minibatches_function(f_n_wrong, X[:len(VX)],
                                                         Z[:len(VZ)])*len(VX)
            info.update({
                'time': passed,
                'val_emp': f_wrong_val,
                'train_emp': f_wrong_train
            })
            row = '%(n_iter)i\t%(val_loss)g\t%(time)g\t%(train_emp)g\t%(' \
                  'val_emp)g' % info
            print row
            if (i % 5) == 0:
                np.save(self.model_name, info['best_pars'])
        np.save(self.model_name, info['best_pars'])
        m.parameters.data[:] = info['best_pars']
        return m

if __name__ == "__main__":
    cw = Cifar100Worker()
    cw.run()