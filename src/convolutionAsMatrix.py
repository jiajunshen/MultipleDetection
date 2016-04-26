import numpy as np
import lasagne
import theano.tensor as T
import theano

# this function is to transfer a convolution operation into a matrix multiplication
# For example, an image with dimension of 3x28x28 X convoluted by filter of 3x5x5 F, originally it can be written as
# X (*) F
# We are trying to write this as a matrix multiplication form as following:
# Source: X': flatten version of X with size 784 (28 * 28).
# Destination: X_f: flatten version of convoluted X with size 576 (24 * 24) [valid convolution] or 1024 (32 * 32) [full convolution]
# Find matrix of dimension  576 x 784 (1024 x 784) T, so that X_f = T %*% X'
# Remember here: the filter flip is important as the default setting for filters in lasagne is flipped

def findMatrix(convfilter, convolutionType = "full", originalSize = 28, flip_filters = True):
    if flip_filters:
        convfilter[:,:,:,:] = convfilter[:,:,:,::-1]
        convfilter[:,:,:,:] = convfilter[:,:,::-1,:]    

    numOfFilters, numOfChannels, filterSize = convfilter.shape[:3]

    resultMatrix = np.zeros(((originalSize + filterSize - 1) * (originalSize + filterSize - 1) * numOfFilters, originalSize * originalSize * numOfChannels))

    for i in range(resultMatrix.shape[0]):
        target_coordinate = np.unravel_index(i, (numOfFilters, originalSize + filterSize - 1, originalSize + filterSize - 1))
        target_xy_cooridinate = [target_coordinate[1], target_coordinate[2]]

        for j in range(originalSize * originalSize):
            source_coordinate = np.unravel_index(j, (originalSize, originalSize))
            source_xy_coordinate = [sum(x) for x in zip(tuple((source_coordinate[0], source_coordinate[1])), tuple((filterSize-1, filterSize - 1)))]
            diff_xy_coordinate = np.array(source_xy_coordinate) - np.array(target_xy_cooridinate)

            if(diff_xy_coordinate[0]<filterSize and diff_xy_coordinate[0] >= 0 and diff_xy_coordinate[1] <filterSize and diff_xy_coordinate[1] >= 0):
                addConvFilter = convfilter[target_coordinate[0]]
                modifyValue = addConvFilter[:,diff_xy_coordinate[0], diff_xy_coordinate[1]]
                modifyCoordinate = np.array([np.arange(numOfChannels), np.ones(numOfChannels) * source_coordinate[0], np.ones(numOfChannels) * source_coordinate[1]], dtype = np.int64)
                flattenIndex = np.ravel_multi_index(modifyCoordinate, (numOfChannels, originalSize, originalSize))
                if (np.mean(resultMatrix[i, flattenIndex]!=0)):
                    print("=====?=======")
                    print resultMatrix[i, flattenIndex]
                    print("============")

                resultMatrix[i, flattenIndex] = modifyValue


    return resultMatrix


def test_accuracy(data, filters):
    originalImage = data
    input_var = T.tensor4('input variable') 
    network = lasagne.layers.InputLayer(shape = data.shape, input_var = input_var)
    network = lasagne.layers.Conv2DLayer(network, num_filters = filters.shape[0], filter_size = filters.shape[-1], pad = 'full', W = filters, b = lasagne.init.Constant(0.), flip_filters = True)
    output = lasagne.layers.get_output(network, input_var)
    output_function = theano.function([input_var], output)
    return output_function(originalImage) 



if __name__ == "__main__":
    testFilter = np.array(np.zeros((1,1,2,2)), dtype = np.float32)
    testFilter[0,0,0,0] = 1
    testFilter[0,0,1,1] = 1
    #testFilter[0,0,1,1] = 1
    #testFilter[0,0,2,2] = 1
    testFilter = np.array(np.random.random((1, 1, 2, 2)), dtype = np.float32)
    data = np.array(np.ones((1,1,4,4)), dtype = np.float32)
    
    convResult = test_accuracy(data, testFilter)
    
    transformMatrix = findMatrix(testFilter, originalSize = 4).T
    
    #print(transformMatrix)
    print(testFilter)
    #print(np.mean((np.dot(data.reshape(1, 1 * 4 * 4), transformMatrix) - convResult.reshape(1,-1))**2))
    print(convResult.reshape(1,-1))
    print(np.dot(data.reshape(1, 1 * 4 *4), transformMatrix))
