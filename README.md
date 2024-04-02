# ArtCNN

## Overview
These are Super-Resolution Convolutional Neural Networks as GLSL shaders for mpv. They implement a simple feed-forward architecture with one long-skip connection and a pixel-shuffle layer to get the HR image.

![Model Architecture](./Images/model_architecture.png "Model Architecture")

The main variant of the shader is offered in 3 sizes, these are meant to "respect" the source and generate fairly neutral outputs:
- `C4F64`: This has 4 internal convolution layers with 64 filters each.
- `C4F32`: This has 4 internal convolution layers with 32 filters each.
- `C4F16`: This has 4 internal convolution layers with 16 filters each.

## Technical Details
The models are trained on an anime dataset containing screenshot from the following shows:
- Violet Evergarden
- Koe no Katachi
- Kimi no Na Wa
- Hibike Euphonium
- Yuru Camp
- Sword Art Online Progressive

The screenshots were split into smaller 256x256 patches and downsampled with the `box` filter.
The L1 loss function was used alongside the Adam optimiser.
