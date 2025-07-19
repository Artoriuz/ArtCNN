import cv2
import numpy as np


def load_image(input, grayscale=True):
    input = cv2.imread(input, cv2.IMREAD_COLOR)

    if grayscale:
        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY, 0)
        input = np.expand_dims(input, axis=-1)
    else:
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB, 0)

    input = np.array(input).astype(np.float32) / 255.0
    input = np.clip(input, 0.0, 1.0)
    return input


def save_image(output, filename, grayscale=True):
    output = np.squeeze(output)
    output = np.around(output * 255.0).astype(np.uint8)

    if not grayscale:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR, 0)

    cv2.imwrite(filename, output)
