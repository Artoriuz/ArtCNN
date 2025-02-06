# ArtCNN

## Overview
ArtCNN is a collection of simple SISR CNNs aimed at anime content.

Two distinct architectures are currently offered:
- `R`: Bigger model aimed mostly at non real-time tasks like rescaling. On top of having more filters per convolution layer, the model was also made much deeper with the help of residual blocks and short-skip connections. Offered only in the ONNX format.
- `C`: Original ArtCNN models optimised mostly for speed. These should only be used for real-time tasks like video playback. The architecture consists of a series of convolution layers aided by a single long-skip connection. Offered in the ONNX format and as GLSL shaders.

4 sizes are currently offered:
- `R16F96`: This has 16 residual blocks and 96 filters per convolution layer. Should generally give you the best reconstruction quality. ~4m params.
- `R8F64`: This has 8 residual blocks and 64 filters per convolution layer. An attempt at balancing quality and performance for non real-time tasks. ~926k params.
- `C4F32`: This has 4 internal convolution layers with 32 filters each. Use this on real-time tasks if your system can handle it. ~48k params.
- `C4F16`: This has 4 internal convolution layers with 16 filters each. Cheaper variant that should work well on most modern GPUs. ~12k params.

Regarding the suffixes:
- Models without any suffixes are the baseline. These are neutral luma doublers.
- `DS` variants are trained to denoise and sharpen, which is usually useful for most web sources.
- `Chroma` variants are trained to reconstruct chroma. These are intended to be used on 4:2:0 content and will not work correctly in any other scenario.

You may occasionaly find some experimental models under the `Experiments` directory.

## mpv Instructions
Add something like this to your mpv config:
```
vo=gpu-next
glsl-shader="path/to/shader/ArtCNN_C4F16_DS.glsl"
```

## VapourSynth Instructions
ArtCNN is natively supported by [vs-mlrt](https://github.com/AmusementClub/vs-mlrt). Please follow the instructions found there.

Alternatively, can also run the GLSL shaders with [vs-placebo](https://github.com/Lypheo/vs-placebo).

## Examples
![ArtCNN Example](./Images/artcnn_examples.png "ArtCNN Example")
