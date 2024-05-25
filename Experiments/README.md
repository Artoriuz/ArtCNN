# ArtCNN Experiments

## RB8F96
This model is roughly based on [EDSR](https://arxiv.org/abs/1707.02921) and [SPAN](https://arxiv.org/abs/2311.12770). It consists of:
- 8 Residual Blocks composed by 3 convolution layers and a short-skip connection (conv->relu->conv->relu->conv->add).
- 96 filters in each convolutional layer.
- Channel concatenation for the long-skip connection.
- Pixel-shuffling.

## C24F96
This model has the exact same architecture as the normal ArtCNN models, the only difference is in its size. It was trained to evaluate whether the architectural changes in RB8F96 are actually helping, and the answer seems to be yes.
- The number of filters and convolutions were chosen to roughly match RB8F96.
