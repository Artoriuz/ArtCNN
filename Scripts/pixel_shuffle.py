import numpy as np
import cv2

def depth_to_space(input_array, block_size):
    height, width, depth = input_array.shape
    assert depth % (block_size ** 2) == 0, "Depth must be divisible by block_size squared."

    new_depth = depth // (block_size ** 2)
    new_height = height * block_size
    new_width = width * block_size

    input_array = input_array.reshape(height, width, block_size, block_size, new_depth)
    input_array = input_array.transpose(0, 2, 1, 3, 4)
    output_array = input_array.reshape(new_height, new_width, new_depth)

    return output_array

def space_to_depth(input_array, block_size):
    height, width, depth = input_array.shape
    assert height % block_size == 0, "Height must be divisible by block_size."
    assert width % block_size == 0, "Width must be divisible by block_size."

    new_height = height // block_size
    new_width = width // block_size
    new_depth = depth * (block_size ** 2)

    input_array = input_array.reshape(new_height, block_size, new_width, block_size, depth)
    input_array = input_array.transpose(0, 2, 1, 3, 4)
    output_array = input_array.reshape(new_height, new_width, new_depth)

    return output_array

block_size = 2

input = cv2.imread("input.png", cv2.IMREAD_COLOR)
input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY, 0)
input = np.expand_dims(input, axis=-1)

outputs = space_to_depth(input, block_size)
cv2.imwrite("output_0.png", outputs[:, :, 0])
cv2.imwrite("output_1.png", outputs[:, :, 1])
cv2.imwrite("output_2.png", outputs[:, :, 2])
cv2.imwrite("output_3.png", outputs[:, :, 3])
