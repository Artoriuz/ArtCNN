import numpy as np
import cv2
from onnxconverter_common import float16


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


def rgb2ycbcr_bt601(image):
    """
    Convert RGB image to YCbCr following BT.601 standard.

    Args:
    Image (numpy.ndarray): RGB image with channels last.

    Returns:
    Image (numpy.ndarray): YCbCr image with channels last.
    """
    y = np.dot(image, np.array([0.299, 0.587, 0.114]))
    cb = np.dot(image, np.array([-0.168736, -0.331264, 0.5])) + 0.5
    cr = np.dot(image, np.array([0.5, -0.418688, -0.081312])) + 0.5

    ycbcr_image = np.stack((y, cb, cr), axis=-1, dtype=np.float32)
    ycbcr_image = np.clip(ycbcr_image, 0.0, 1.0)

    return ycbcr_image


def ycbcr2rgb_bt601(image):
    """
    Convert YCbCr image to RGB following BT.601 standard.

    Args:
    Image (numpy.ndarray): YCbCr image with channels last.

    Returns:
    Image (numpy.ndarray): RGB image with channels last.
    """
    image[:, :, 1] = image[:, :, 1] - 0.5
    image[:, :, 2] = image[:, :, 2] - 0.5

    r = np.dot(image, np.array([1.0, 0.0, 1.402]))
    g = np.dot(image, np.array([1.0, -0.344136, -0.714136]))
    b = np.dot(image, np.array([1.0, 1.772, 0.0]))

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
    return tuple(np.unstack(image, axis=-1))


def stack(*channels):
    squeezed_channels = [np.squeeze(ch) for ch in channels]
    image = np.stack(squeezed_channels, axis=-1)
    return image


def self_ensemble_augment(image):
    rotations = [0, 90, 180, 270]
    images = []

    for rotation in rotations:
        rotated_image = np.rot90(image.copy(), rotation // 90)
        images.append(rotated_image)

    image = cv2.flip(image, 1)

    for rotation in rotations:
        rotated_image = np.rot90(image.copy(), rotation // 90)
        images.append(rotated_image)

    return images


def self_ensemble_combine(images):
    assert len(images) == 8, "Expected 8 images"
    deaugmented = []

    for i, rotation in enumerate([0, 90, 180, 270]):
        img = images[i].copy()
        img = np.rot90(img, -rotation // 90)
        deaugmented.append(np.squeeze(img))

    for i, rotation in enumerate([0, 90, 180, 270]):
        img = images[i + 4].copy()
        img = np.rot90(img, -rotation // 90)
        img = cv2.flip(img, 1)
        deaugmented.append(np.squeeze(img))

    deaugmented = np.array(deaugmented)
    image = np.mean(deaugmented, axis=0, keepdims=False)
    return image


def convert_to_fp16(model_fp32):
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    return model_fp16
