# ArtCNN

## Overview
ArtCNN is a collection of SISR CNNs optimised for anime content.

Two distinct architectures are currently offered:
- `C`: Original ArtCNN models optimised mostly for speed. The architecture consists of a series of convolution layers aided by a single long-skip connection. Mostly offered as GLSL shaders to be used in real-time on mpv.
- `R`: Bigger model aimed mostly at encoding tasks. On top of having more filters per convolution layer, the model was also made much deeper with the help of residual blocks and short-skip connections. These are only offered in the ONNX format.

The `R` architecture is offered in 2 sizes:
- `R16F96`: This has 16 residual blocks and 96 filters per convolution layer. Should generally give you the best reconstruction quality.
- `R8F64`: This has 8 residual blocks and 64 filters per convolution layer. An attempt at balancing quality and performance.

The `C` architecture is offered in 4 sizes:
- `C16F64`: This has 16 internal convolution layers with 64 filters each. Offered only in the ONNX format as a faster alternative to the `R` models.
- `C4F32`: This has 4 internal convolution layers with 32 filters each. You need a relatively decent GPU to run this well.
- `C4F16`: This has 4 internal convolution layers with 16 filters each. You should be able to run this on most modern GPUs.
- `C4F8`: This has 4 internal convolution layers with 8 filters each. You should probably only use this on very slow systems.

Regarding the suffixes:
- Models without any suffixes are the baseline. These are neutral luma doublers.
- `DS` variants are trained to denoise and sharpen, which is usually useful for most web sources.
- `CMP` variants are compute shaders. These are generally much faster on modern GPUs, specially on Vulkan. You should always try to run these variants first, and only fallback to the fragment shaders if you encounter problems and/or performance issues.
- `Chroma` variants are trained to reconstruct chroma. These are intended to be used on 4:2:0 content and will not work as intended in any other scenario (which means you can't use them after luma doublers).

You may occasionaly find some models under the "Experiments" directory. This is meant to serve as a testing ground for future models.

## Training Details
The luma models are trained on an anime dataset containing images from the following sources:
- Violet Evergarden
- Koe no Katachi
- Kimi no Na Wa
- Hibike Euphonium (Chikai and Ensemble)
- Yuru Camp (Film only)
- SAO (OS and Progressive)
- Evangelion: 3.0+1.0

The Chroma models are trained on DIV2K+Manga109.

The images are split into smaller 256x256 patches and downsampled with the box filter.
The L1 loss function is used alongside the Adam optimiser.
The models are trained using Keras 3 with its JAX backend.

## mpv Instructions
Add something like this to your mpv config:
```
glsl-shader="path/to/shader/ArtCNN_C4F16_DS_CMP.glsl"
```

## VapourSynth Instructions
ArtCNN is natively supported by [vs-mlrt](https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py). Please follow the instructions found there.

Alternatively, can also run the GLSL shaders with [vs-placebo](https://github.com/Lypheo/vs-placebo).

ONNX models should be generally preferred over their GLSL counterparts for both quality and performance reasons.
