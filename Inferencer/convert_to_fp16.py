import onnx
import glob
from pathlib import Path
from onnxconverter_common import float16

filelist = sorted(glob.glob("*.onnx"))

for model in filelist:
    model = Path(model)
    model_fp32 = onnx.load(model)
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    onnx.save(model_fp16, f"{model.stem}_fp16.onnx")
