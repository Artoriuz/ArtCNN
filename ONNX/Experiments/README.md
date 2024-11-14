# ArtCNN_R8F64_4x
Same as the normal R8F64 but 4x instead of 2x.

# ArtCNN_R8F64_LL
Same as the normal R8F64 but trained with images downsampled in linear light. In practice, this outputs slightly thicker lineart and can be a bit better if you plan on downscaling in linear light afterwards.

# ArtCNN_R8F64_SAFE
Same as the normal R8F64 but trained with less information density. I'm calling this `SAFE` because it tries less hard to produce fine-details, which means it's also less likely to be too wrong or to produce temporal instability between similarish frames. On most of my test images the baseline model does perform better, but this is consistently better at the images the baseline model struggles with.

# ArtCNN_R16F86_SAFE
Self explanatory.
