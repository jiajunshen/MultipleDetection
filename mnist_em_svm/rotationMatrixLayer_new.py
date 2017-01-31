import lasagne
import theano
import theano.tensor as T

class RotationMatrixLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n, Degree = lasagne.init.Normal(5), Translation = lasagne.init.Normal(0.01), **kwargs):
        super(RotationMatrixLayer, self).__init__(incoming, **kwargs)
        self.n = n
        self.Translation = self.add_param(Translation, (n, ), name = "Translation")
        self.Degree = self.add_param(Degree, (n, ), name = "Degree")
        print()

    def get_output_for(self, input, **kwargs):
        cosT = T.cos(self.Degree * 3.1415926 / 180.0)
        sinT = T.sin(self.Degree * 3.1415926 / 180.0)
        # zeros = T.zeros_like(cosT)
        zeros = self.Translation
        finalResult = T.stack([cosT, sinT, zeros, -sinT, cosT, zeros], axis = 1)
        return finalResult

    def get_output_shape_for(self, input_shape):
        return (self.n, 6)
