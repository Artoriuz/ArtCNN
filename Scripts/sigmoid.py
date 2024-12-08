import numpy as np
import cv2

def sigmoid(x, slope, center):
    offset = 1.0 / (1.0 + np.exp(slope * center))
    scale  = 1.0 / (1.0 + np.exp(slope * (center - 1.0))) - offset
    return center - np.log(1.0 / (x * scale + offset) - 1.0) * 1.0 / slope

input = cv2.imread('meme.png', cv2.IMREAD_COLOR)
input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY, 0)
input = np.array(input).astype(np.float32) / 255.0
input = np.power(input, 2.2)
input = np.clip(input, 0.0, 1.0)

output = sigmoid(input, 10.5, 0.5)
output = output * 255.0
output = np.squeeze((np.around(output)).astype(np.uint8))

cv2.imwrite('sigmoid.png', output)
