from theano.compile import  MonitorMode

__author__ = 'kosklain'

from theano.tensor.signal import downsample
import theano
import numpy
from theano import tensor as T


def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]

def print_mat(matrix):
    for row in matrix:
        string = "|"
        for element in row:
            string += " "+str(element)
        print string+" |"
print " "


input = T.dtensor4('input')
maxpool_shape = (4, 4)
maxpool_stride = (2, 2)
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True, stride=maxpool_stride)
f = theano.function([input],pool_out)
theano.printing.debugprint(f.maker.fgraph.outputs[0])
invals = numpy.random.RandomState(1).rand(2, 2, 9, 9)
print 'With ignore_border set to True:'



print 'invals[0, 0, :, :] =\n', print_mat(invals[0, 1, :, :])
print 'output[0, 0, :, :] =\n', print_mat(f(invals)[0, 1, :, :])

pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=False, stride=maxpool_stride)
f = theano.function([input],pool_out)
theano.printing.debugprint(f.maker.fgraph.outputs[0])
print 'With ignore_border set to False:'
print 'invals[1, 0, :, :] =\n ', print_mat(invals[1, 0, :, :])
print 'output[1, 0, :, :] =\n ', print_mat(f(invals)[1, 0, :, :])