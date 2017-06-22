import lasagne
import theano
import theano.tensor as T
from collections import OrderedDict

def transform_affine(thetaX, thetaY, thetaZ, input):
    num_batch, num_channels, height, width, depth = input.shape

    # Three rotation matrices for three axes
    thetaX = T.reshape(thetaX, (-1, 4, 4))
    thetaY = T.reshape(thetaY, (-1, 4, 4))
    thetaZ = T.reshape(thetaZ, (-1, 4, 4))


    # Three rotation matrices multiplied together
    theta = T.batched_dot(T.batched_dot(thetaX, thetaY), thetaZ)

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = T.cast(height, 'int64')
    out_width = T.cast(width, 'int64')
    out_depth = T.cast(depth, 'int64')
    grid = meshgrid(out_height, out_width, out_depth)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    # x_s, y_s, z_s shape (n, out_height*out_width*out_depth)
    # grid is (4, out_height*out_width*out_depth)
    T_g = T.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    z_s = T_g[:, 2]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()
    z_s_flat = z_s.flatten()

    # dimshuffle input to  (bs, height, width, depth, channels)
    input_dim = input.dimshuffle(0, 2, 3, 4, 1)
    input_transformed = interpolate(
        input_dim, x_s_flat, y_s_flat,z_s_flat,
        out_height, out_width, out_depth)

    output = T.reshape(
        input_transformed, (num_batch, out_height, out_width, out_depth, num_channels))
    output = output.dimshuffle(0, 4, 1, 2, 3)  # dimshuffle to conv format
    return output

def interpolate(im, x, y, z, out_height, out_width, out_depth):
    # *_f are floats
    num_batch, height, width, depth, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)
    depth_f = T.cast(width, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)
    z = T.clip(z, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height/depth - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)
    z = (z + 1) / 2 * (depth_f - 1)


    # obtain indices of the 2x2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    z0_f = T.floor(z)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    z1_f = z0_f + 1
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    z0 = T.cast(z0_f, 'int64')
    x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
    y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')
    z1 = T.cast(T.minimum(z1_f, depth_f - 1), 'int64')

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width*depth, channels]. We need
    # to offset all indices to match the flat version
    dim3 = depth
    dim2 = depth * width
    dim1 = depth * width * height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, out_height*out_width*out_depth)
    base_x0 = x0*dim3
    base_x1 = x1*dim3
    base_y0 = y0*dim2
    base_y1 = y1*dim2
    idx_a = base + base_x0 + base_y0 + z0
    idx_b = base + base_x0 + base_y0 + z1
    idx_c = base + base_x0 + base_y1 + z0
    idx_d = base + base_x0 + base_y1 + z1
    idx_e = base + base_x1 + base_y0 + z0
    idx_f = base + base_x1 + base_y0 + z1
    idx_g = base + base_x1 + base_y1 + z0
    idx_h = base + base_x1 + base_y1 + z1

    # use indices to lookup pixels for all samples
    # For 3d, we need 8 points
    # a:(0,0,0); b:(0,0,1); c:(0,1,0); d:(0,1,1);
    # e:(1,0,0); f:(1,0,1); g:(1,1,0); h:(1,1,1);
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]
    Ie = im_flat[idx_e]
    If = im_flat[idx_f]
    Ig = im_flat[idx_g]
    Ih = im_flat[idx_h]

    # distance ratio
    xd = (x - x0_f).dimshuffle(0, 'x')
    yd = (y - y0_f).dimshuffle(0, 'x')
    zd = (z - z0_f).dimshuffle(0, 'x')

    # interpolate along x
    c00 = Ia * (1 - xd) + Ie * xd
    c01 = Ic * (1 - xd) + Ig * xd
    c10 = Ib * (1 - xd) + If * xd
    c11 = Id * (1 - xd) + Ih * xd

    # interpolate along z
    c0 = c00 * (1 - zd) + c10 * zd
    c1 = c01 * (1 - zd) + c11 * zd

    # interpolate along y
    c = c0 * (1 - yd) + c1 * yd

    return c

def meshgrid(height, width, depth):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(T.dot(T.ones((height, 1)),
                      linspace(-1.0, 1.0, width).dimshuffle('x', 0)).reshape((-1, 1)),
                T.ones((1, depth)))

    y_t = T.dot(linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width * depth)))

    z_t = T.dot(T.ones((height * width, 1)), linspace(-1.0, 1.0, depth).dimshuffle('x', 0))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    z_t_flat = z_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, z_t_flat, ones], axis=0)
    return grid

def linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


class AffineTransformation3DLayer(lasagne.layers.Layer):
    # incoming would be n x C x W x H
    def __init__(self, incoming, n, name=None, DegreeX=lasagne.init.Normal(5), DegreeY = lasagne.init.Normal(5),
                 DegreeZ = lasagne.init.Normal(5), Scaling = lasagne.init.Normal(0.01), **kwargs):
        super(AffineTransformation3DLayer, self).__init__(incoming, **kwargs)
        self.n = n
        self.get_output_kwargs = []
        self.DegreeX = self.add_param(DegreeX, (n, ), name="DegreeX")
        self.DegreeY = self.add_param(DegreeY, (n, ), name="DegreeY")
        self.DegreeZ = self.add_param(DegreeZ, (n, ), name="DegreeZ")


    def get_output_for(self, input, **kwargs):
        #self.translation = T.zeros((n, 2))
        #ones = T.exp(self.scaling[:, 0])
        ones = T.ones((self.n,))
        zeros = T.zeros((self.n,))
        cosTx = T.cos(self.DegreeX * 3.1415926 / 180.0)
        sinTx = T.sin(self.DegreeX * 3.1415926 / 180.0)
        cosTy = T.cos(self.DegreeY * 3.1415926 / 180.0)
        sinTy = T.sin(self.DegreeY * 3.1415926 / 180.0)
        cosTz = T.cos(self.DegreeZ * 3.1415926 / 180.0)
        sinTz = T.sin(self.DegreeZ * 3.1415926 / 180.0)
        thetaX = T.stack([ones, zeros, zeros, zeros, zeros, cosTx, -sinTx, zeros, zeros, sinTx, cosTx, zeros, zeros, zeros, zeros, ones], axis = 1)
        thetaY = T.stack([cosTy, zeros, sinTy, zeros, zeros, ones, zeros, zeros, -sinTy, zeros, cosTy, zeros, zeros, zeros, zeros, ones], axis = 1)
        thetaZ = T.stack([cosTz, -sinTz, zeros, zeros, sinTz, cosTz, zeros, zeros, zeros, zeros, ones, zeros, zeros, zeros, zeros, ones], axis = 1)

        return transform_affine(thetaX, thetaY, thetaZ, input)

    def get_output_shape_for(self, input_shape):
        return input_shape
