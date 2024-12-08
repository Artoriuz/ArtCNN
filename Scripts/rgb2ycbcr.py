import cv2
import numpy as np

def rgb2ycbcr_bt709(image):
    """
    Convert RGB image to YCbCr following BT.709 standard.

    Args:
    Image (numpy.ndarray): RGB image with channels last.

    Returns:
    Image (numpy.ndarray): YCbCr image with channels last.
    """
    y  = np.dot(image, np.array([0.2126, 0.7152, 0.0722]))
    cb = np.dot(image, np.array([-0.114572, -0.385428, 0.5])) + 0.5
    cr = np.dot(image, np.array([0.5, -0.454153, -0.045847])) + 0.5

    ycbcr_image = np.stack((y, cb, cr), axis=-1)
    ycbcr_image = np.clip(ycbcr_image, 0.0, 1.0)

    return ycbcr_image

input = cv2.imread('input.png', cv2.IMREAD_COLOR)
input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB, 0)
input = input.astype(np.float32) / 255.0
ycbcr_image = rgb2ycbcr_bt709(input)
ycbcr_image = np.squeeze((np.around(ycbcr_image * 255.0)).astype(np.uint8))
ycbcr_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_RGB2BGR, 0)
cv2.imwrite('output.png', ycbcr_image)
