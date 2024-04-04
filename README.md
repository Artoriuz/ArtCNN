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
- SAO Films

The images were split into smaller 256x256 patches and downsampled with the box filter.

The L1 loss function was used alongside the Adam optimiser.

## Benchmarks
`aoko.png`:
| Shader/Filter       | MAE      | PSNR    | SSIM   | MS-SSIM |   | MAE (N) | PSNR (N) | SSIM (N) | MS-SSIM (N) |   | Mean   |
|---------------------|----------|---------|--------|---------|---|---------|----------|----------|-------------|---|--------|
| ArtCNN_C4F64        | 5.23E-03 | 35.9756 | 0.9901 |  0.9994 |   |  1.0000 |   1.0000 |   1.0000 |      1.0000 |   | 1.0000 |
| ArtCNN_C4F32        | 5.48E-03 | 35.6177 | 0.9893 |  0.9994 |   |  0.9830 |   0.9650 |   0.9878 |      0.9969 |   | 0.9832 |
| ArtCNN_C4F16        | 6.39E-03 | 34.4153 | 0.9860 |  0.9992 |   |  0.9217 |   0.8475 |   0.9379 |      0.9808 |   | 0.9220 |
| FSRCNNX_x2_16-0-4-1 | 9.75E-03 | 31.7201 | 0.9758 |  0.9976 |   |  0.6944 |   0.5841 |   0.7825 |      0.8289 |   | 0.7225 |
| FSRCNNX_x2_8-0-4-1  | 1.08E-02 | 31.0438 | 0.9705 |  0.9975 |   |  0.6258 |   0.5180 |   0.7017 |      0.8196 |   | 0.6663 |
| ravu-lite-ar-r4     | 1.09E-02 | 30.4805 | 0.9681 |  0.9974 |   |  0.6177 |   0.4629 |   0.6651 |      0.8117 |   | 0.6393 |
| ravu-lite-ar-r3     | 1.12E-02 | 30.2648 | 0.9666 |  0.9974 |   |  0.5969 |   0.4418 |   0.6418 |      0.8101 |   | 0.6226 |
| ravu-zoom-ar-r3     | 1.11E-02 | 30.0447 | 0.9684 |  0.9972 |   |  0.6025 |   0.4203 |   0.6699 |      0.7907 |   | 0.6209 |
| ravu-lite-ar-r2     | 1.14E-02 | 29.7235 | 0.9660 |  0.9975 |   |  0.5824 |   0.3889 |   0.6330 |      0.8187 |   | 0.6057 |
| ravu-zoom-ar-r2     | 1.15E-02 | 29.6350 | 0.9661 |  0.9970 |   |  0.5764 |   0.3803 |   0.6347 |      0.7664 |   | 0.5894 |
| lanczos             | 1.53E-02 | 28.3051 | 0.9497 |  0.9962 |   |  0.3216 |   0.2503 |   0.3844 |      0.6945 |   | 0.4127 |
| polar_lanczossharp  | 1.61E-02 | 27.8741 | 0.9448 |  0.9949 |   |  0.2635 |   0.2082 |   0.3101 |      0.5760 |   | 0.3395 |
| bilinear            | 2.00E-02 | 25.7442 | 0.9245 |  0.9889 |   |  0.0000 |   0.0000 |   0.0000 |      0.0000 |   | 0.0000 |
