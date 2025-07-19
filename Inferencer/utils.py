import numpy as np
import cv2


def rgb2ycbcr(image):
    """
    Convert RGB image to YCbCr following BT.709 standard.

    Args:
    Image (numpy.ndarray): RGB image with channels last.

    Returns:
    Image (numpy.ndarray): YCbCr image with channels last.
    """
    y = np.dot(image, np.array([0.2126, 0.7152, 0.0722]))
    cb = np.dot(image, np.array([-0.114572, -0.385428, 0.5])) + 0.5
    cr = np.dot(image, np.array([0.5, -0.454153, -0.045847])) + 0.5

    ycbcr_image = np.stack((y, cb, cr), axis=-1, dtype=np.float32)
    ycbcr_image = np.clip(ycbcr_image, 0.0, 1.0)

    return ycbcr_image


def ycbcr2rgb(image):
    """
    Convert YCbCr image to RGB following BT.709 standard.

    Args:
    Image (numpy.ndarray): YCbCr image with channels last.

    Returns:
    Image (numpy.ndarray): RGB image with channels last.
    """
    image[:, :, 1] = image[:, :, 1] - 0.5
    image[:, :, 2] = image[:, :, 2] - 0.5

    r = np.dot(image, np.array([1.0, 0.0, 1.5748]))
    g = np.dot(image, np.array([1.0, -0.1873, -0.4681]))
    b = np.dot(image, np.array([1.0, 1.8556, 0.0]))

    rgb_image = np.stack((r, g, b), axis=-1, dtype=np.float32)
    rgb_image = np.clip(rgb_image, 0.0, 1.0)

    return rgb_image


def bilinear(image, scaling_factor):
    image = cv2.resize(
        image,
        None,
        fx=scaling_factor,
        fy=scaling_factor,
        interpolation=cv2.INTER_LINEAR_EXACT,
    )
    return image


def unstack(image):
    image = np.squeeze(image)
    x, y, z = np.unstack(image, axis=-1)
    return (x, y, z)


def stack(x, y, z):
    x = np.squeeze(x)
    y = np.squeeze(y)
    z = np.squeeze(z)
    image = np.stack((x, y, z), axis=-1)
    return image
