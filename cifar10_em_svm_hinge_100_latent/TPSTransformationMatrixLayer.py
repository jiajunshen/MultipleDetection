import lasagne
import theano
import theano.tensor as T
from collections import OrderedDict
from lasagne.utils import as_tuple
import numpy as np
def interpolate(im, x, y, out_height, out_width):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
    y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output

def meshgrid(height, width):
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
    x_t = T.dot(T.ones((height, 1)),
                linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid

def linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start

def get_transformed_points_tps(new_points, source_points, coefficients,
                                num_points, batch_size):
    """
    Calculates the transformed points' value using the provided coefficients
    :param new_points: num_batch x 2 x num_to_transform tensor
    :param source_points: 2 x num_points array of source points
    :param coefficients: coefficients (should be shape (num_batch, 2,
        control_points + 3))
    :param num_points: the number of points
    :return: the x and y coordinates of each transformed point. Shape (
        num_batch, 2, num_to_transform)
    """

    # Calculate the U function for the new point and each source point as in
    # ref [2]
    # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
    # squared distance

    # Calculate the squared dist between the new point and the source points
    to_transform = new_points.dimshuffle(0, 'x', 1, 2)
    stacked_transform = T.tile(to_transform, (1, num_points, 1, 1))
    r_2 = T.sum(((stacked_transform - source_points.dimshuffle(
            'x', 1, 0, 'x')) ** 2), axis=2)

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
    log_r_2 = T.log(r_2)
    distances = T.switch(T.isnan(log_r_2), r_2 * log_r_2, 0.)

    # Add in the coefficients for the affine translation (1, x, and y,
    # corresponding to a_1, a_x, and a_y)
    upper_array = T.concatenate([T.ones((batch_size, 1, new_points.shape[2]),
                                        dtype=theano.config.floatX),
                                 new_points], axis=1)
    right_mat = T.concatenate([upper_array, distances], axis=1)

    # Calculate the new value as the dot product
    new_value = T.batched_dot(coefficients, right_mat)
    return new_value

def transform_thin_plate_spline(
        dest_offsets, input, right_mat, L_inv, source_points, out_height,
        out_width, precompute_grid, downsample_factor):

    num_batch, num_channels, height, width = input.shape
    num_control_points = source_points.shape[1]

    # reshape destination offsets to be (num_batch, 2, num_control_points)
    # and add to source_points
    dest_points = source_points + T.reshape(
            dest_offsets, (num_batch, 2, num_control_points))

    # Solve as in ref [2]
    coefficients = T.dot(dest_points, L_inv[:, 3:].T)

    if precompute_grid:

        # Transform each point on the source grid (image_size x image_size)
        right_mat = T.tile(right_mat.dimshuffle('x', 0, 1), (num_batch, 1, 1))
        transformed_points = T.batched_dot(coefficients, right_mat)

    else:

        # Transformed grid
        out_height = T.cast(height // downsample_factor[0], 'int64')
        out_width = T.cast(width // downsample_factor[1], 'int64')
        orig_grid = meshgrid(out_height, out_width)
        orig_grid = orig_grid[0:2, :]
        orig_grid = T.tile(orig_grid, (num_batch, 1, 1))

        # Transform each point on the source grid (image_size x image_size)
        transformed_points = get_transformed_points_tps(
                orig_grid, source_points, coefficients, num_control_points,
                num_batch)

    # Get out new points
    x_transformed = transformed_points[:, 0].flatten()
    y_transformed = transformed_points[:, 1].flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = interpolate(
            input_dim, x_transformed, y_transformed,
            out_height, out_width)

    output = T.reshape(input_transformed,
                       (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output



def U_func_numpy(x1, y1, x2, y2):
    """
    Function which implements the U function from Bookstein paper
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: value of z
    """

    # Return zero if same point
    if x1 == x2 and y1 == y2:
        return 0.

    # Calculate the squared Euclidean norm (r^2)
    r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Return the squared norm (r^2 * log r^2)
    return r_2 * np.log(r_2)

def initialize_tps(num_control_points, input_shape, downsample_factor,
                    precompute_grid):
    """
    Initializes the thin plate spline calculation by creating the source
    point array and the inverted L matrix used for calculating the
    transformations as in ref [2]_
    :param num_control_points: the number of control points. Must be a
        perfect square. Points will be used to generate an evenly spaced grid.
    :param input_shape: tuple with 4 elements specifying the input shape
    :param downsample_factor: tuple with 2 elements specifying the
        downsample for the height and width, respectively
    :param precompute_grid: boolean specifying whether to precompute the
        grid matrix
    :return:
        right_mat: shape (num_control_points + 3, out_height*out_width) tensor
        L_inv: shape (num_control_points + 3, num_control_points + 3) tensor
        source_points: shape (2, num_control_points) tensor
        out_height: tensor constant specifying the ouptut height
        out_width: tensor constant specifying the output width
    """

    # break out input_shape
    _, _, height, width = input_shape

    # Create source grid
    grid_size = np.sqrt(num_control_points)
    x_control_source, y_control_source = np.meshgrid(
        np.linspace(-1, 1, grid_size),
        np.linspace(-1, 1, grid_size))

    # Create 2 x num_points array of source points
    source_points = np.vstack(
            (x_control_source.flatten(), y_control_source.flatten()))

    # Convert to floatX
    source_points = source_points.astype(theano.config.floatX)

    # Get number of equations
    num_equations = num_control_points + 3

    # Initialize L to be num_equations square matrix
    L = np.zeros((num_equations, num_equations), dtype=theano.config.floatX)

    # Create P matrix components
    L[0, 3:num_equations] = 1.
    L[1:3, 3:num_equations] = source_points
    L[3:num_equations, 0] = 1.
    L[3:num_equations, 1:3] = source_points.T

    # Loop through each pair of points and create the K matrix
    for point_1 in range(num_control_points):
        for point_2 in range(point_1, num_control_points):

            L[point_1 + 3, point_2 + 3] = U_func_numpy(
                    source_points[0, point_1], source_points[1, point_1],
                    source_points[0, point_2], source_points[1, point_2])

            if point_1 != point_2:
                L[point_2 + 3, point_1 + 3] = L[point_1 + 3, point_2 + 3]

    # Invert
    L_inv = np.linalg.inv(L)

    if precompute_grid:
        # Construct grid
        out_height = np.array(height // downsample_factor[0]).astype('int64')
        out_width = np.array(width // downsample_factor[1]).astype('int64')
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, out_width),
                               np.linspace(-1, 1, out_height))
        ones = np.ones(np.prod(x_t.shape))
        orig_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        orig_grid = orig_grid[0:2, :]
        orig_grid = orig_grid.astype(theano.config.floatX)

        # Construct right mat

        # First Calculate the U function for the new point and each source
        # point as in ref [2]
        # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
        # squared distance
        to_transform = orig_grid[:, :, np.newaxis].transpose(2, 0, 1)
        stacked_transform = np.tile(to_transform, (num_control_points, 1, 1))
        stacked_source_points = \
            source_points[:, :, np.newaxis].transpose(1, 0, 2)
        r_2 = np.sum((stacked_transform - stacked_source_points) ** 2, axis=1)

        # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
        log_r_2 = np.log(r_2)
        log_r_2[np.isinf(log_r_2)] = 0.
        distances = r_2 * log_r_2

        # Add in the coefficients for the affine translation (1, x, and y,
        # corresponding to a_1, a_x, and a_y)
        upper_array = np.ones(shape=(1, orig_grid.shape[1]),
                              dtype=theano.config.floatX)
        upper_array = np.concatenate([upper_array, orig_grid], axis=0)
        right_mat = np.concatenate([upper_array, distances], axis=0)

        # Convert to tensors
        out_height = T.as_tensor_variable(out_height)
        out_width = T.as_tensor_variable(out_width)
        right_mat = T.as_tensor_variable(right_mat)

    else:
        out_height = None
        out_width = None
        right_mat = None

    # Convert to tensors
    L_inv = T.as_tensor_variable(L_inv)
    source_points = T.as_tensor_variable(source_points)

    return right_mat, L_inv, source_points, out_height, out_width

class TPSTransformationMatrixLayer(lasagne.layers.Layer):
    # incoming would be n x C x W x H 
    def __init__(self, incoming, n, W = lasagne.init.Normal(5), name=None, downsample_factor = 1, control_points=16, precompute_grid = 'auto', **kwargs):
        super(TPSTransformationMatrixLayer, self).__init__(
                incoming, **kwargs)
        self.n = n
        self.W = self.add_param(W, (n, control_points * 2), name = "localization")


        self.downsample_factor = as_tuple(downsample_factor, 2)
        self.control_points = control_points

        input_shp = self.input_shape

        if round(np.sqrt(control_points)) != np.sqrt(
                control_points):
            raise ValueError("The number of control points must be"
                             " a perfect square.")

        if len(input_shp) != 4:
            raise ValueError("The input network must have a 4-dimensional "
                             "output shape: (batch_size, num_input_channels, "
                             "input_rows, input_columns)")

        # Process precompute grid
        can_precompute_grid = all(s is not None for s in input_shp[2:])
        if precompute_grid == 'auto':
            precompute_grid = can_precompute_grid
        elif precompute_grid and not can_precompute_grid:
            raise ValueError("Grid can only be precomputed if the input "
                             "height and width are pre-specified.")
        self.precompute_grid = precompute_grid

        # Create source points and L matrix
        self.right_mat, self.L_inv, self.source_points, self.out_height, \
            self.out_width = initialize_tps(
                control_points, input_shp, self.downsample_factor,
                precompute_grid)

    def get_output_shape_for(self, input_shapes):
        shape = input_shapes
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s // f)
                                  for s, f in zip(shape[2:], factors)))

    def get_output_for(self, input, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        # Get input and destination control points
        dest_offsets = self.W
        return transform_thin_plate_spline(
                dest_offsets, input, self.right_mat, self.L_inv,
                self.source_points, self.out_height, self.out_width,
                self.precompute_grid, self.downsample_factor)
