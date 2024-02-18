# ArtCNN

## Overview
These are Super-Resolution Convolutional Neural Networks as GLSL shaders for mpv. They implement a simple feed-forward architecture with one long-skip connection and a pixel-shuffle layer to get the HR image.

![Model Architecture](./Images/model_architecture.png "Model Architecture")

The main variant of the shader is offered in 3 sizes, these are meant to "respect" the source and generate fairly neutral outputs:
- `C4F32`: This has 4 internal convolution layers with 32 filters each. This is the "big" variant of the shader. This is the best version of the shader for very high quality content, but it is a bit resource-intensive for real-time video playback and you should only consider using it if you have a good GPU.
- `C4F16`: This has 4 internal convolution layers with 16 filters each. This is the "normal" variant of the shader. Most semi-recent discrete GPUs should be able to handle this.
- `C4F8`: This has 4 internal convolution layers with 8 filters each. This is the "small" variant of the shader. You should only use this for performance or power consumption reasons.

A few other variants are also offered, these are meant to cover specific needs:
- `LL`: Trained with images downsampled in linear light. Use this if you suspect the content has been downsampled in linear light.
- `SH`: Trained with LR images that have been downsampled with Hermite. Use this if you want some mild sharpening.
- `DN`: Trained with JPEG LR images that have been moderately compressed. Use this to clean compression artifacts at the expense of some fine-detail.
- `DS`: Trained with JPEG LR images that have been moderately compressed and downsampled with Hermite. This provides mild artifact cleaning and sharpening, which might work well for low quality web sources.

If you plan on using ArtCNN for fractional scaling factors below 2x, the `SH` and `DS` variants might work better with mpv's default `dscale` filter (`hermite`).

The chroma variants are a bit experimental and currently they simply reuse the luma models to upscale the chroma channels separately. This works to some extent but it's probably not ideal.

## Technical Details
The shaders are trained on the Manga109 dataset using the Adam optimiser with a learning rate of 1e-4 and the L1/MAE loss function. The high-resolution images are downsampled with a box filter, and they're also split into small 64x64 patches for performance and memory reasons.

You can check the `ArtCNN_Training.ipynb` Colab Notebook for details.

## Instructions
Add something like this to your mpv config:
```
vo=gpu-next
glsl-shader="path/to/shader/ArtCNN_C4F16.glsl"
```

## Example
![Example](./Images/example.png "Example")
