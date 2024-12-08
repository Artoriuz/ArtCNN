import numpy as np
import cv2

def desigmoid(x, slope, center):
    offset = 1.0 / (1.0 + np.exp(slope * center))
    scale  = 1.0 / (1.0 + np.exp(slope * (center - 1.0))) - offset
    return (1.0 / (1.0 + np.exp(slope * (center - x))) - offset) * 1.0 / scale

input = cv2.imread('sigmoid_2x.png', cv2.IMREAD_COLOR)
input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY, 0)
input = np.array(input).astype(np.float32) / 255.0

output = desigmoid(input, 10.5, 0.5)
output = np.power(output, 1.0 / 2.2)
output = np.clip(output, 0.0, 1.0)
output = output * 255.0
output = np.squeeze((np.around(output)).astype(np.uint8))

cv2.imwrite('desigmoid.png', output)
