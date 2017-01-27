import lasagne
import theano
import theano.tensor as T
from collections import OrderedDict


class SelectLayer(lasagne.layers.Layer):
    # incoming would be n x C x W x H 
    def __init__(self, incoming, n, **kwargs):
        super(SelectLayer, self).__init__(incoming, **kwargs)
        loc_shp = self.input_shape
        if (n != loc_shp[1]):
           raise ValueError("n does not equal to the dimension of loc_shp")
        self.n = n

    def get_output_for(self, input, **kwargs):
        theta = T.reshape(theta, (-1, 2, 3))
        input_reshape = T.reshape(input, (-1, self.n, self.n))
        input_reshape = input_reshape.dimshuffle(1, 2, 0)
        target_index = T.eye(self.n, self.n)
        input_reshape = input_reshape[target_index.nonzero()]
        return input_reshape.dimshuffle(1, 0)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] // self.n, input_shape[1])
