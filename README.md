# ArtCNN

## Overview
Super-Resolution Convolutional Neural Networks as GLSL shaders for mpv. ArtCNN implements a simple feed-forward architecture with one long-skip connection and a pixel-shuffle layer to get the HR image.

![Model Architecture](./Images/model_architecture.png "Model Architecture")

The model is offered in 4 sizes:
- `C16F64`: This has 16 internal convolution layers with 64 filters each. Offered only in the ONNX format. If you're interested in using ArtCNN outside of mpv you should probably use this.
- `C4F32`: This has 4 internal convolution layers with 32 filters each. You need a relatively decent GPU to run this well. Also offered in the ONNX format.
- `C4F16`: This has 4 internal convolution layers with 16 filters each. You should be able to run this on most modern GPUs.
- `C4F8`: This has 4 internal convolution layers with 8 filters each. You should probably only use this on very slow systems.

Regarding the suffixes:
- Shaders without any suffixes are the base models. These are meant to respect the source and produce fairly neutral outputs.
- Shaders with the `DS` suffix are trained to denoise and sharpen, which is usually useful for most web sources.
- Shaders with the `CMP` suffix are compute shaders. These are still experimental, but they're usually faster (specially on Vulkan).
- The old `Chroma`, `YCbCr` and `RGB` variants can be found under the "Old" directory. These have not been updated to reflect the new software stack and training dataset yet.

## Technical Details
The models were trained on an anime dataset containing screenshots from the following shows:
- Violet Evergarden
- Koe no Katachi
- Kimi no Na Wa
- Hibike Euphonium
- Yuru Camp
- SAO OS and Progressive

The images were split into smaller 256x256 patches and downsampled with the box filter.
The L1 loss function was used alongside the Adam optimiser.
The models were trained using Keras 3 and its JAX backend.

## Instructions
Add something like this to your mpv config:
```
glsl-shader="path/to/shader/ArtCNN_C4F16_DS.glsl"
```

## Example
![Example](./Images/example.png "Example")
