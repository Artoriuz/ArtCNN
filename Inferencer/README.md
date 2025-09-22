# Inferencer
`inferencer.py` is a simple CLI tool that wraps ONNX Runtime.

## Instructions
```shell
usage: inferencer.py [-h] [-m MODEL] [-cm CHROMA_MODEL] [-t {luma,rgb,ycbcr,denoise}] [--self-ensemble] input

ONNX Inferencer

positional arguments:
  input                 Input image or directory

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        ArtCNN Model
  -cm CHROMA_MODEL, --chroma-model CHROMA_MODEL
                        ArtCNN Chroma Model
  -t {luma,rgb,ycbcr,denoise}, --task {luma,rgb,ycbcr,denoise}
                        Task to perform
  --self-ensemble       Enable self-ensemble inference
  ```

## Examples

> [!IMPORTANT]
> The examples given below assume the ONNX models are in the same directory.

Luma upscaling using a standard luma model:
```shell
python inferencer.py image.png --model ArtCNN_R16F96.onnx --task luma
```

RGB upscaling using a standard luma model on each channel:
```shell
python inferencer.py image.png --model ArtCNN_R16F96.onnx --task rgb
```

YCbCr upscaling using a luma model and a chroma model together:
```shell
python inferencer.py image.png --model ArtCNN_R16F96.onnx --chroma-model ArtCNN_R16F96_Chroma.onnx --task ycbcr
```

JPEG denoising using a JPEG model:
```shell
python inferencer.py image.jpg --model ArtCNN_R8F64_JPEG420.onnx --task denoise
```

> [!TIP]
> The Execution Provider can be changed from DirectML to TensorRT or MIGraphX for better performance on supported systems.
