# Experiments

## R16F96B
Same as R16F96 but without biases. The idea is forcing the model to be luminance-equivariant which prevent overfitting and potentially improve temporal stability.

This idea comes from these papers, which implement it for denoising models:
https://arxiv.org/pdf/1906.05478
https://arxiv.org/pdf/2008.13751

Whether or not this makes sense for super-resolution is still unclear. My tests have been inconclusive.

## R8F64
Same as the normal R16F96 but much smaller. With only 25% as many weights, this model is intrinsically incapable of extracting as much information from the dataset.
In practice, this leads to better temporal stability because the model becomes way less sensitive to noise.

This obviously comes with a reconstruction quality penalty in compariso to the bigger sibling, but the difference might be too small for it to matter depending on the input.
