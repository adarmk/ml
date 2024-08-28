import numpy as np

kernel_side = 3
stride = 1

pool_side = 2

l1_filters = 32

def relu(x):
    return np.maximum(0, x)

def conv2d(input: np.ndarray, kernels: np.ndarray, biases: np.ndarray):
    assert input.shape[2] == kernels.shape[2], f"Input and kernel depth must match. Input shape: {input.shape}, Kernels shape: {kernels.shape}"
    assert kernels.shape[3] == biases.shape[0], f"Number of kernels must match number of biases. Kernels shape: {kernels.shape}, Biases shape: {biases.shape}"
    
    result = np.zeros((input.shape[0] // stride, input.shape[1] // stride, kernels.shape[3]))

    for i in range(-1, input.shape[0] - 1, stride):
        for j in range(-1, input.shape[1] - 1, stride):
            for kernel in range(kernels.shape[3]):
                kernel_sum = 0
                for k in range(input.shape[2]):
                    for l in range(kernel_side):
                        for m in range(kernel_side):
                            # This is done for padding
                            row_idx = i + l
                            col_idx = j + m
                            if row_idx < 0 or col_idx < 0 or row_idx >= input.shape[0] or col_idx >= input.shape[1]:
                                value = 0
                            else:
                                value = input[row_idx][col_idx][k]
                            kernel_sum += kernels[l][m][k][kernel] * value
                result[i + 1, j + 1, kernel] = kernel_sum + biases[kernel]
                            
    return result
                    

# Test conv2d function
def test_conv2d():
    # Create a simple 5x5x1 input
    input_data = np.array([
        [[1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0]]
    ]).transpose(1, 2, 0)  # Transpose to make it 5x5x1

    # Create a 3x3x1x1 kernel (1 filter)
    # Create a 3x3x1x1 kernel (1 filter)
    kernel = np.array([
        [[[1], [0], [0]],
        [[0], [1], [0]],
        [[0], [0], [0]]]
    ]).transpose(1 ,2 ,0 ,3) # Transpose to make it 3x3x1x1

    # Create a bias
    bias = np.array([0])

    # Apply convolution
    result = conv2d(input_data, kernel, bias)

    # Print results
    print("Input:")
    print(input_data[:,:,0])
    print("\nKernel:")
    print(kernel[:,:,0,0])
    print("\nConvolution result:")
    print(result[:,:,0])

# Run the test
test_conv2d()










