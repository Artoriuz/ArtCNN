import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

# Load your ONNX model using DirectML
models_dir = Path("E:/Code/artcnn/ONNX")
model = "ArtCNN_R16F96.onnx"

model = models_dir / model
session = ort.InferenceSession(model, providers=["DmlExecutionProvider"])

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print("Input name:", input_name)
print("Input shape:", input_shape)
print("Input type:", input_type)

# Load input
input = cv2.imread("./downscaled.png", cv2.IMREAD_COLOR)
input = np.array(input).astype(np.float32) / 255.0
input = np.clip(input, 0.0, 1.0)

(input_b, input_g, input_r) = cv2.split(input)

input_b = np.expand_dims(input_b, axis=0)
input_b = np.expand_dims(input_b, axis=0)

input_g = np.expand_dims(input_g, axis=0)
input_g = np.expand_dims(input_g, axis=0)

input_r = np.expand_dims(input_r, axis=0)
input_r = np.expand_dims(input_r, axis=0)

# Run inference
pred_b = session.run(None, {input_name: input_b})
print("Output:", pred_b)
pred_g = session.run(None, {input_name: input_g})
print("Output:", pred_g)
pred_r = session.run(None, {input_name: input_r})
print("Output:", pred_r)


# Save output
pred_b = np.array(pred_b)
pred_g = np.array(pred_g)
pred_r = np.array(pred_r)
pred_bgr = np.stack((pred_b, pred_g, pred_r), axis=-1)
pred_bgr = np.squeeze(pred_bgr)

pred_bgr = pred_bgr * 255.0
pred_bgr = np.around(pred_bgr).astype(np.uint8)
cv2.imwrite(f"./ArtCNN_R16F96_RGB.png", pred_bgr)
