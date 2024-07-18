# Experiments

## R16F96B
Same as R16F96 but without biases. The idea is forcing the model to be luminance-equivariant which prevent overfitting and potentially improve temporal stability.

This idea comes from these papers, which implement it for denoising models:
https://arxiv.org/pdf/1906.05478
https://arxiv.org/pdf/2008.13751

Whether or not this makes sense for super-resolution is still unclear. My tests have been inconclusive.
