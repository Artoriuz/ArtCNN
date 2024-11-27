# ArtCNN_R8F64_4x
Same as the normal R8F64 but 4x instead of 2x.

# ArtCNN_R8F64_LL
Same as the normal R8F64 but trained with images downsampled in linear light. In practice, this outputs slightly thicker lineart and can be a bit better if you plan on downscaling in linear light afterwards.

# ArtCNN_R8F64_UNSAFE
Same as the normal R8F64 but trained with more information density. This has the potential to create cleaner and more detailed images, but when it goes wrong it generally produces way worse results.

# ArtCNN_R16F86_UNSAFE
Self explanatory.

# ArtCNN_R8F64_FANART
Same as the normal R8F64 but trained with fanart. Looks a bit better on very clean anime scenes (Hibike S3 for example), but might be too sharp/aggressive for most anime shows.

# ArtCNN_R16F96_FANART
Self explanatory.

# ArtCNN_R8F64_SE
Same as the normal R8F64 but with squeeze and excitation (channel attention) in the residual blocks.

# ArtCNN_R8F64_BF
Same as the normal R8F64 but without biases ("BF" stands for "bias-free").

# ArtCNN_R8F64_AA
Same as the normal R8F64 but trained with images downsampled with point to add aliasing. This model shifts the image.

# ArtCNN_R8F64_DEV
Same as the normal R8F64 but trained with both anime and fanart (approximately 3/4 anime and 1/4 fanart).
