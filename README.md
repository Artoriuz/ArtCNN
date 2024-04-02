# ArtCNN

## Overview
These are Super-Resolution Convolutional Neural Networks as GLSL shaders for mpv. They implement a simple feed-forward architecture with one long-skip connection and a pixel-shuffle layer to get the HR image.

![Model Architecture](./Images/model_architecture.png "Model Architecture")

The main variant of the shader is offered in 3 sizes, these are meant to "respect" the source and generate fairly neutral outputs:
- `C4F64`: This has 4 internal convolution layers with 64 filters each.
- `C4F32`: This has 4 internal convolution layers with 32 filters each.
- `C4F16`: This has 4 internal convolution layers with 16 filters each.

## Technical Details
The models were trained on an anime dataset containing screenshots from the following shows:
- Violet Evergarden
- Koe no Katachi
- Kimi no Na Wa
- Hibike Euphonium
- Yuru Camp
- Sword Art Online Progressive

The images were split into smaller 256x256 patches and downsampled with the box filter.

The L1 loss function was used alongside the Adam optimiser.

## Benchmarks
`aoko.png`
| Shader/Filter       | MAE      | PSNR    | SSIM   | MS-SSIM |   | MAE (N) | PSNR (N) | SSIM (N) | MS-SSIM (N) |   | Mean   |
|---------------------|----------|---------|--------|---------|---|---------|----------|----------|-------------|---|--------|
| ArtCNN_C4F64        | 5.34E-03 | 35.8962 | 0.9899 |  0.9994 |   |  1.0000 |   1.0000 |   1.0000 |      1.0000 |   | 1.0000 |
| ArtCNN_C4F32        | 5.55E-03 | 35.5938 | 0.9892 |  0.9994 |   |  0.9855 |   0.9702 |   0.9900 |      0.9969 |   | 0.9857 |
| ArtCNN_C4F16        | 6.41E-03 | 34.4313 | 0.9860 |  0.9992 |   |  0.9271 |   0.8557 |   0.9406 |      0.9828 |   | 0.9265 |
| FSRCNNX_x2_16-0-4-1 | 9.75E-03 | 31.7201 | 0.9758 |  0.9976 |   |  0.6997 |   0.5886 |   0.7851 |      0.8300 |   | 0.7259 |
| FSRCNNX_x2_8-0-4-1  | 1.08E-02 | 31.0438 | 0.9705 |  0.9975 |   |  0.6307 |   0.5220 |   0.7040 |      0.8207 |   | 0.6694 |
| ravu-lite-ar-r4     | 1.09E-02 | 30.4805 | 0.9681 |  0.9974 |   |  0.6224 |   0.4665 |   0.6674 |      0.8128 |   | 0.6423 |
| ravu-lite-ar-r3     | 1.12E-02 | 30.2648 | 0.9666 |  0.9974 |   |  0.6015 |   0.4453 |   0.6439 |      0.8112 |   | 0.6255 |
| ravu-zoom-ar-r3     | 1.11E-02 | 30.0447 | 0.9684 |  0.9972 |   |  0.6072 |   0.4236 |   0.6722 |      0.7918 |   | 0.6237 |
| ravu-lite-ar-r2     | 1.14E-02 | 29.7235 | 0.9660 |  0.9975 |   |  0.5869 |   0.3920 |   0.6351 |      0.8198 |   | 0.6084 |
| ravu-zoom-ar-r2     | 1.15E-02 | 29.6350 | 0.9661 |  0.9970 |   |  0.5808 |   0.3833 |   0.6368 |      0.7674 |   | 0.5921 |
| lanczos             | 1.53E-02 | 28.3051 | 0.9497 |  0.9962 |   |  0.3241 |   0.2523 |   0.3856 |      0.6954 |   | 0.4144 |
| polar_lanczossharp  | 1.61E-02 | 27.8741 | 0.9448 |  0.9949 |   |  0.2655 |   0.2098 |   0.3112 |      0.5768 |   | 0.3408 |
| bilinear            | 2.00E-02 | 25.7442 | 0.9245 |  0.9889 |   |  0.0000 |   0.0000 |   0.0000 |      0.0000 |   | 0.0000 |
