# ArtCNN Experiments

## RB16F96
This model is roughly based on [EDSR](https://arxiv.org/abs/1707.02921) and [SPAN](https://arxiv.org/abs/2311.12770). It consists of:
- 16 Residual Blocks composed by 3 convolution layers (2 activations) and a short-skip connection (conv->relu->conv->relu->conv->add).
- 96 filters in each convolutional layer.
- Channel concatenation for the long-skip connection.
- Pixel-shuffling to get the HR image.

## RB16F96_B
Same as RB16F96, but with element-wise addition instead of channel concatenation for the long-skip connection. I think this is the best model on average, but it can lose to the concat version sometimes. In practice this is a very small difference and it shouldn't matter, but concatenating feature maps with identical shapes when we can just add them feels somewhat wrong to me. The idea of concatenating them came from SPAN, but SPAN concatenates feature maps from 4 different layers instead of 2.
