# ArtCNN

## Overview
ArtCNN is a collection of SISR CNNs optimised for anime content.

Two distinct architectures are currently offered:
- `R`: Bigger model aimed mostly at encoding tasks. On top of having more filters per convolution layer, the model was also made much deeper with the help of residual blocks and short-skip connections. Offered in the ONNX format.
- `C`: Original ArtCNN models optimised mostly for speed. The architecture consists of a series of convolution layers aided by a single long-skip connection. Offered as GLSL shaders to be used in real-time on mpv.

The `R` architecture is offered in 2 sizes:
- `R16F96`: This has 16 residual blocks and 96 filters per convolution layer. Should generally give you the best reconstruction quality.
- `R8F64`: This has 8 residual blocks and 64 filters per convolution layer. An attempt at balancing quality and performance.

The `C` architecture is also offered in 2 sizes:
- `C4F32`: This has 4 internal convolution layers with 32 filters each. You need a relatively decent GPU to run this well.
- `C4F16`: This has 4 internal convolution layers with 16 filters each. You should be able to run this on most modern GPUs.

Regarding the suffixes:
- Models without any suffixes are the baseline. These are neutral luma doublers.
- `CMP` variants are compute shaders. These are generally much faster on modern GPUs, specially on Vulkan. You should always try to run these variants first, and only fallback to the fragment shaders if you encounter problems and/or performance issues.
- `DS` variants are trained to denoise and sharpen, which is usually useful for most web sources.
- `Chroma` variants are trained to reconstruct chroma. These are intended to be used on 4:2:0 content and will not work as intended in any other scenario (which means you can't use them after luma doublers).

You may occasionaly find some models under the "Experiments" directories. This is meant to serve as a testing grounds.

## mpv Instructions
Add something like this to your mpv config:
```
glsl-shader="path/to/shader/ArtCNN_C4F16_DS_CMP.glsl"
```

## VapourSynth Instructions
ArtCNN is natively supported by [vs-mlrt](https://github.com/AmusementClub/vs-mlrt). Please follow the instructions found there.

Alternatively, can also run the GLSL shaders with [vs-placebo](https://github.com/Lypheo/vs-placebo).

## Examples
![ArtCNN Example](./Images/artcnn_examples.png "ArtCNN Example")

## FAQ

### What architecture does ArtCNN use?

The original `C` architecture is the result of the research I've conducted during my MSc. Starting from [EDSR](https://arxiv.org/abs/1707.02921), I miniaturised and thoroughly simplified the model to maximise quality within a very constrained performance budget. The subsequent `R` architecture was born from various experiments trying to scale ArtCNN up while still attempting to balance quality and performance.

My goal is to keep ArtCNN as vanilla as possible, so don't expect bleeding-edge ideas to be adopted until they've stood the test of time.

### Why ReLU activations?

Using ReLU instead of fancier options like the GELU or the SiLU is a deliberate choice. The quality gains from switching to different activations are difficult to justify when you take into account the performance penalty. ArtCNN is aimed at video playback and encoding, where speed matters. If you want to understand why the ReLU is faster, feel free to check [ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models](https://arxiv.org/abs/2310.04564).

### Why the L1 loss?

The L1 loss (mean absolute error) is still the standard loss used to train state-of-the-art distortion-based SISR models. Researchers have attempted to design more sophisticated losses to better match human perception of quality, but ultimately the best results are often obtained by [combining the L1 loss with something else](https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf), just to help it get out of local minima. I've been playing around with structural-similarity and frequency-domain losses, but the results so far have not been conclusive.

### Why AdamW?

AdamW's weight decay seems to help with generalisation. Models trained with Adam can often reduce the loss further, but test scores don't reflect that as an improvement. AdamW is also quickly becoming the new standard on recent papers.

### Why 3 conv layers instead of 2 in the residual blocks?

This was an entirely empirical choice as well. The usual `Conv->ReLU->Conv->Add` residual block from [EDSR](https://arxiv.org/abs/1707.02921) ended up slightly worse than `Conv->ReLU->Conv->ReLU->Conv->Add` even when employed on slightly larger models with a learning capacity advantage. I've experimented with deeper residual blocks, but they did not yield consistent improvements. [SPAN](https://arxiv.org/abs/2311.12770) has a similar residual block configuration if we exclude the attention mechanism, and the authors of [NFNet](https://arxiv.org/abs/2102.06171) found it to be an improvement as well.

### Why no channel attention?

The global average pooling layer required for channel attention is very slow.

### Why no vision transformers?

CNNs have a stronger inductive bias to help solve image processing tasks, this means you don't need as much data or as big of a model to get good results.

Papers like [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) and [ConvNets Match Vision Transformers at Scale](https://arxiv.org/abs/2310.16764) have also showed that CNNs are still competitive with transformers even at the scales in which they were designed to excel.

As an electrical engineer I also simply find CNNs more elegant.

### Why Keras?

I'm just familiar with Keras. I've tried migrating to PyTorch a few times, but there was always something annoying enough about it for me to scrap the idea. Keras 3 supports TF, PyTorch and JAX as backends which also makes it very versatile. If I were to migrate away from Keras now, I'd probably just go straight to [Flax](https://flax.readthedocs.io/en/latest/) as that would keep me in the same ecosystem.
