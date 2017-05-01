import lasagne
import theano
import theano.tensor as T
from collections import OrderedDict


class BrightnessAdjustLayer(lasagne.layers.Layer):
    # incoming would be n x C x W x H
    def __init__(self, incoming, n, name=None, Brightness = lasagne.init.Normal(0), **kwargs):
        super(BrightnessAdjustLayer, self).__init__(incoming, **kwargs)
        self.n = n
        self.get_output_kwargs = []
        self.brightness = self.add_param(Brightness, (n, ), name = "brightness_multiplier")

    def get_output_for(self, input, **kwargs):
        v_channel_bias = self.brightness.dimshuffle(0, 'x', 'x',)
        data_channel_v = input[:, 2] + v_channel_bias
        data_channel_v = data_channel_v.clip(0.0, 1.0)
        output = T.stack([input[:,0], input[:,1], data_channel_v], axis = 1)
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape

