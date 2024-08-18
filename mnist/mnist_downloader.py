from keras.datasets import mnist
import numpy as np
import struct

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save training data to text file
with open('mnist_train.txt', 'w') as f:
    for label, img in zip(y_train, x_train):
        img_str = ','.join(str(pixel) for row in img for pixel in row)
        f.write(f'{label} {img_str}\n')

# Save test data to text file
with open('mnist_test.txt', 'w') as f:
    for label, img in zip(y_test, x_test):
        img_str = ','.join(str(pixel) for row in img for pixel in row)
        f.write(f'{label} {img_str}\n')



# # Save training data to binary file
# with open('mnist_train.bin', 'wb') as f:
#     for label, img in zip(y_train, x_train):
#         f.write(struct.pack('B', label))  # Write label as unsigned char (1 byte)
#         f.write(img.tobytes())  # Write image as bytes (28x28 = 784 bytes)

# # Save test data to binary file
# with open('mnist_test.bin', 'wb') as f:
#     for label, img in zip(y_test, x_test):
#         f.write(struct.pack('B', label))  # Write label as unsigned char (1 byte)
#         f.write(img.tobytes())  # Write image as bytes (28x28 = 784 bytes)