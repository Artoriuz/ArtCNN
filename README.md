# ArtCNN

## Overview
Super-Resolution Convolutional Neural Networks as GLSL shaders for mpv. ArtCNN implements a simple feed-forward architecture with one long-skip connection and a pixel-shuffle layer to get the HR image.

![Model Architecture](./Images/model_architecture.png "Model Architecture")

The main variant of the model is offered in 3 sizes, these are meant to "respect" the source and generate fairly neutral outputs:
- `C4F64`: This has 4 internal convolution layers with 64 filters each. Probably too slow on most GPUs.
- `C4F32`: This has 4 internal convolution layers with 32 filters each. You need a relatively decent GPU to run this well.
- `C4F16`: This has 4 internal convolution layers with 16 filters each. You should be able to run this on most modern GPUs.

`DS` variants are also offered. These are meant to denoise and sharpen, which is usually useful on bad sources.

The old `Chroma`, `YCbCr` and `RGB` variants can be found under the "Old" directory. These have not been updated to reflect the new software stack and training dataset yet.

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
The models were trained using Keras 3 with its JAX backend.

## Example
![Example](./Images/example.png "Example")
