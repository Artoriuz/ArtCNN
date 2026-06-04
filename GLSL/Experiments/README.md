# ArtCNN YCbCr Experimental Shaders

## Overview
These shaders are trained to double YCbCr frames taking chroma subsampling into account. They're supposed to be used with `--cscale=bilinear` for a fully neutral look, but filters like mitchell work well too.

- `YCbCr`: Simply upscales.
- `YCbCr_DN`: Upscales while taking noise into account. Denoises.
- `YCbCr_DS`: Upscales while taking noise and blur into account. Denoises and sharpens.
