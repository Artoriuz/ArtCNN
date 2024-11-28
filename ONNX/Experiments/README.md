# 4x
Same as the normal models but 4x instead of 2x.

# LL
Same as the normal models but trained with images downsampled in linear light. In practice, this outputs slightly thicker lineart and can be a bit better if you plan on downscaling in linear light afterwards.

# FANART
Same as the normal models but trained with fanart. Looks a bit better on very clean anime scenes (Hibike S3 for example), but might be too sharp/aggressive for most anime shows.

# SE
Same as the normal models but with squeeze and excitation (channel attention) in the residual blocks.

# BF
Same as the normal models but without biases ("BF" stands for "bias-free").

# AA
Same as the normal models but trained with images downsampled with point to add aliasing. Shifts the image.

# DEV
Same as the normal models but trained with both anime and fanart (approximately 3/4 anime and 1/4 fanart).
