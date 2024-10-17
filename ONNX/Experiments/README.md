# ArtCNN_R8F64_4x
Same as the normal R8F64 but 4x instead of 2x.

# ArtCNN_R8F64_SAFE
Same as the normal R8F64 but trained with less information density. I'm calling this `SAFE` because it tries less hard to produce fine-details, which means it's also less likely to be too wrong or to produce temporal instability between similarish frames. On most of my test images the baseline model does perform better, but this is consistently better at the images the baseline model struggles with.

# ArtCNN_R8F64_SiLU
Same as the normal R8F64 but with [SiLU activations](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html). In general the SiLU is supposed to be slightly better than the ReLU, but it's also a bit slower. As it stands, this model is often better than the baseline R8F64, but this is not always true. If anything it seems to have converged to a more general solution, with fewer bad outliers.
