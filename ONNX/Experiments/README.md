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

# SAFE
Same as the normal models but trained with anime frames only (normal models have some fanarts in the dataset as well).

# DCT
Same as the normal models but trained with the difference between the DCTs as the loss function.

# DFT
Same as the normal models but trained with the difference between the DFTs as the loss function.

# MSSIM
Same as the normal models but trained with MAE+MS-SSIM as the loss function.

# MSMAE
Same as the normal models but trained with MAE at various scales as the loss function.

# DCT_MAE
Same as the normal models but trained with MAE+DCT as the loss function.

# V2
Uses the RCAB from RCAN instead of normal residual blocks.

# DEINT
1x deinterlacer model that receives even lines and outputs odd lines.
