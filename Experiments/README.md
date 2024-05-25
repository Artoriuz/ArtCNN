# ArtCNN Experiments

## RB8F96
This model is roughly based on [EDSR](https://arxiv.org/abs/1707.02921) and [SPAN](https://arxiv.org/abs/2311.12770). It consists of:
- 8 Residual Blocks composed by 3 convolution layers and a short-skip connection (conv->relu->conv->relu->conv->add).
- 96 filters in each convolutional layer.
- Channel concatenation for the long-skip connection.
- Pixel-shuffling.
