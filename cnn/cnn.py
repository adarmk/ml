import numpy as np

kernel_side = 3
stride = 1

pool_side = 2
pool_stride = 2

l1_filters = 32

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def softmax_prime(x):
    s = softmax(x)
    return s * (1 - s)


def conv2d(input: np.ndarray, kernels: np.ndarray, biases: np.ndarray) -> np.ndarray:
    assert input.shape[2] == kernels.shape[2], f"Input and kernel depth must match. Input shape: {input.shape}, Kernels shape: {kernels.shape}"
    assert kernels.shape[3] == biases.shape[0], f"Number of kernels must match number of biases. Kernels shape: {kernels.shape}, Biases shape: {biases.shape}"
    
    output = np.zeros((input.shape[0] // stride, input.shape[1] // stride, kernels.shape[3]))

    # Assumes both kernel sidelengths are odd
    padded_input = np.pad(input, pad_width=((kernels.shape[0]//2, kernels.shape[1]//2),(kernels.shape[0]//2,kernels.shape[1]//2),(0,0)), mode='constant', constant_values=0)

    for i in range(0, output.shape[0]):
            for j in range(output.shape[1]):
                kernel_input = padded_input[i*stride:i*stride + kernels.shape[0], j*stride:j*stride + kernels.shape[1], :]
                product = kernel_input[:, :, :, np.newaxis] * kernels
                output[i, j, :] = np.sum(product, axis=(0,1,2)) + biases
    return output

# Assumes first two dimensions of `input` are divisible by `pool_side`
def pool2d(input: np.ndarray) -> np.ndarray:
    output_shape = (input.shape[0] // pool_side, input.shape[1] // pool_side, input.shape[2])
    strided = np.lib.stride_tricks.sliding_window_view(input, (pool_side, pool_side, input.shape[2]))[::pool_stride, ::pool_stride]
    return np.max(strided, axis=(3,4))

# TODO: implement dropout
def feed_forward(
    img: np.ndarray, 
    params_l1: tuple[np.ndarray, np.ndarray], 
    params_l2: tuple[np.ndarray, np.ndarray], 
    params_l3: tuple[np.ndarray, np.ndarray], 
    params_l4: tuple[np.ndarray, np.ndarray], 
    inference=False
):
    (kernels, biases) = params_l1
    z1 = conv2d(img, kernels, biases)
    a1 = relu(z1)
    pooled_a1 = pool2d(a1)

    (kernels, biases) = params_l2
    z2 = conv2d(pooled_a1, kernels, biases)
    a2 = relu(z2)
    pooled_a2 = pool2d(a2)

    flattened_a2 = pooled_a2.flatten()

    (weights, biases) = params_l3 

    z3 = np.dot(weights, flattened_a2) + biases 
    a3 = relu(z3)

    (weights, biases) = params_l4
    z4 = np.dot(weights, a3)
    a4 = softmax(z4)

    if inference:
        return a4
    else:
        return z1, a1, pooled_a1, z2, a2, pooled_a2, flattened_a2, z3, a3, z4, a4


















