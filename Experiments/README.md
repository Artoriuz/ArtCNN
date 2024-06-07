# ArtCNN Experiments

## RB8F96
This model is roughly based on [EDSR](https://arxiv.org/abs/1707.02921) and [SPAN](https://arxiv.org/abs/2311.12770). It consists of:
- 8 Residual Blocks composed by 3 convolution layers and a short-skip connection (conv->relu->conv->relu->conv->add).
- 96 filters in each convolutional layer.
- Channel concatenation for the long-skip connection.
- Pixel-shuffling.

## RCAB8F96
Same as RB8F96 but with the channel attention mechanisms from [RCAN](https://arxiv.org/abs/1807.02758).

## RB16F9
Pretty much the same as RB8F96, but twice as long.
- Trained to check whether making the model even bigger is feasible.
- Shows signs of overfitting, dataset might need to either be expanded or augmented.

## C24F96
This model has the exact same architecture as the normal ArtCNN models, the only difference is in its size. It was trained to evaluate whether the architectural changes in the experiments are actually helping, and the answer seems to be yes.
- The number of filters and convolutions were chosen to roughly match RB8F96/RCAB8F96.
