import cv2
import numpy as np

def ycbcr2rgb_bt709(image):
    """
    Convert YCbCr image to RGB following BT.709 standard.

    Args:
    Image (numpy.ndarray): YCbCr image with channels last.

    Returns:
    Image (numpy.ndarray): RGB image with channels last.
    """
    image[:, :, 1] = image[:, :, 1] - 0.5
    image[:, :, 2] = image[:, :, 2] - 0.5

    r  = np.dot(image, np.array([1.0, 0.0, 1.5748]))
    g = np.dot(image, np.array([1.0, -0.1873, -0.4681]))
    b = np.dot(image, np.array([1.0, 1.8556, 0.0]))

    rgb_image = np.stack((r, g, b), axis=-1)
    rgb_image = np.clip(rgb_image, 0.0, 1.0)

    return rgb_image

input = cv2.imread('input.png', cv2.IMREAD_COLOR)
input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB, 0)
input = input.astype(np.float32) / 255.0
rgb_image = ycbcr2rgb_bt709(input)
rgb_image = np.squeeze((np.around(rgb_image * 255.0)).astype(np.uint8))
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR, 0)
cv2.imwrite('output.png', rgb_image)
