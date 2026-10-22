# ArtCNN YCbCr Experimental Shaders

## Overview
These shaders are trained to double YCbCr frames taking chroma subsampling into account. They're supposed to be used with `--cscale=bilinear` for a fully neutral look, but you can also use other filters.

- `YCbCr`: Simply upscales.
- `YCbCr_DN`: Upscales while taking noise, scaling and sharpening artifacts into account. Tuned to produce a soft output.
- `YCbCr_DS`: Upscales while taking noise, scaling and sharpening artifacts into account. Tuned to produce a sharp output.
