# Inferencer
`inferencer.py` is a simple CLI tool that wraps ONNX Runtime.

## Instructions
```shell
usage: inferencer.py [-h] [-m MODEL] [-cm CHROMA_MODEL] [-t TASK] input

ONNX Inferencer

positional arguments:
  input                 Input image

options:
  -h, --help            show this help message and exit
  -m, --model MODEL     ArtCNN Model
  -cm, --chroma-model CHROMA_MODEL
                        ArtCNN Chroma Model
  -t, --task TASK       Task to perform
  ```

## Examples
Luma upscaling using a standard luma model:
```shell
python Inferencer/inferencer.py image.png --model ONNX/ArtCNN_R16F96.onnx --task luma
```

RGB upscaling using a standard luma model on each channel:
```shell
python Inferencer/inferencer.py image.png --model ONNX/ArtCNN_R16F96.onnx --task rgb
```

YCbCr upscaling using a luma model and a chroma model together:
```shell
python Inferencer/inferencer.py image.png --model ONNX/ArtCNN_R16F96.onnx --chroma-model ONNX/ArtCNN_R16F96_Chroma.onnx --task ycbcr
```
