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

ArtCNN is trained on a combination of high-quality anime frames and fanart. Models trained only on anime or fanart can be found under the `Flavours` directory. These might be useful depending on what you're upscaling.

You may occasionaly find some models under the `Experiments` directory. This is meant to serve as a testing grounds.

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

## FAQ

### Why these architectures?

The original `C` architecture is the result of the research I've conducted during my MSc. Starting from [EDSR](https://arxiv.org/abs/1707.02921), I miniaturised and thoroughly simplified the model to maximise quality within a very constrained performance budget. The subsequent `R` architecture was born from various experiments trying to scale ArtCNN up while still attempting to balance quality and performance.

My goal is to keep ArtCNN as vanilla as possible, so don't expect bleeding-edge ideas to be adopted until they've stood the test of time.

### Why ReLU activations?

Using ReLU instead of fancier options like the GELU or the SiLU is a deliberate choice. The quality gains from switching to different activations are difficult to justify when you take into account the performance penalty. ArtCNN is aimed at video playback and encoding, where speed matters. If you want to understand why the ReLU is faster, feel free to check [ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models](https://arxiv.org/abs/2310.04564).

### Why the L1 loss?

The L1 loss (mean absolute error) is still the standard loss used to train state-of-the-art distortion-based SISR models. Researchers have attempted to design more sophisticated losses to better match human perception of quality, but ultimately the best results are often obtained by combining the L1 loss with something else, just to help it get out of local minima. This is very well detailed in [Loss Functions for Image Restoration with Neural Networks](https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf).

I've been playing around with structural-similarity and frequency-domain losses, but the results so far have not been conclusive.

### Why AdamW?

AdamW's weight decay seems to help with generalisation. Models trained with Adam can often reduce the loss further, but test scores don't reflect that as an improvement. AdamW is also quickly becoming the new standard on recent papers.

### Why this residual block design?

This was an entirely empirical choice as well. The usual `Conv->ReLU->Conv->Add` residual block from [EDSR](https://arxiv.org/abs/1707.02921) ended up slightly worse than `Conv->ReLU->Conv->ReLU->Conv->Add` even when employed on slightly larger models with a learning capacity advantage. I've experimented with deeper residual blocks, but they did not yield consistent improvements. [SPAN](https://arxiv.org/abs/2311.12770) has a similar residual block configuration if we exclude the attention mechanism, and the authors of [NFNet](https://arxiv.org/abs/2102.06171) found it to be an improvement as well.

I've also experimented with bottlenecked residual blocks and inverted residuals. However, the `1x1` convolution layers introduce additional sequential dependencies, slowing down the model even when the total parameter count remains similar.

### Why depth to space?

The depth to space operation is generally better than using transposed convolutions and it's the standard on SISR models in general. ArtCNN is also designed to have all of its convolution layers operating with LR feature maps, only upsampling them as the very last step. This is mostly done for speed, but it also provides great memory footprint benefits.

### Why no attention mechanisms?

The global average pooling layer required for channel attention is often very slow. Spatial attention offers some benefits but it isn't super convincing. ArtCNN is also small enough that I'm somewhat convinced it doesn't really need attention mechanisms to do well.

### Why no vision transformers?

CNNs have a stronger inductive bias to help solve image processing tasks, this means you don't need as much data or as big of a model to get good results.

Papers like [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) and [ConvNets Match Vision Transformers at Scale](https://arxiv.org/abs/2310.16764) have also showed that CNNs are still competitive with transformers even at the scales in which they were designed to excel. This is also true against Mamba, see: [MambaOut: Do We Really Need Mamba for Vision?](https://arxiv.org/abs/2405.07992).

As an electrical engineer I also simply find CNNs more elegant.

### Why Keras?

I'm just familiar with Keras. I learnt a lot reading [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.google.com/books/edition/_/HHetDwAAQBAJ) and back when I started PyTorch wasn't as dominant in academia as it is today.

I've actually tried migrating to PyTorch a few times, but there was always something annoying enough about it for me to scrap the idea. The biggest deal breaker is the insistence on channels-first when it is not only counter-intuitive but also slower than channels-last. The PyTorch version of R8F64 trains roughly ~40% slower than its Keras+JAX counterpart (though, admittedly, I'm not super familiar with PyTorch so maybe there's a way to speed it up).

Talking about [JAX](https://jax.readthedocs.io/en/latest/index.html), I honestly think it's one of the coolest pieces of software I've ever played with and I'd love to fully migrate to [Flax](https://flax.readthedocs.io/en/latest/) once it can export ONNX natively.
