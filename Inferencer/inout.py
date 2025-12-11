import numpy as np
from PIL import Image


def load_image(input, mode="L"):
    input = Image.open(input).convert(mode)
    profile = input.info.get("icc_profile")
    input = np.array(input)

    if mode == "L":
        input = np.expand_dims(input, axis=-1)

    input = input.astype(np.float32) / 255.0
    input = np.clip(input, 0.0, 1.0)
    return input, profile


def save_image(output, filename, mode="L", icc_profile=None):
    output = np.squeeze(output)
    output = np.around(output * 255.0).astype(np.uint8)
    output = Image.fromarray(output, mode=mode)
    output.save(filename, icc_profile=icc_profile)
