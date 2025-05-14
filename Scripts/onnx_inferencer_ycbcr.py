import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

def rgb2ycbcr_bt709(image):
    """
    Convert RGB image to YCbCr following BT.709 standard.

    Args:
    Image (numpy.ndarray): RGB image with channels last.

    Returns:
    Image (numpy.ndarray): YCbCr image with channels last.
    """
    y  = np.dot(image, np.array([0.2126, 0.7152, 0.0722]))
    cb = np.dot(image, np.array([-0.114572, -0.385428, 0.5])) + 0.5
    cr = np.dot(image, np.array([0.5, -0.454153, -0.045847])) + 0.5

    ycbcr_image = np.stack((y, cb, cr), axis=-1, dtype=np.float32)
    ycbcr_image = np.clip(ycbcr_image, 0.0, 1.0)

    return ycbcr_image

def ycbcr2rgb_bt709(image):
    """
    Convert YCbCr image to RGB following BT.709 standard.

    Args:
    Image (numpy.ndarray): YCbCr image with channels last.

    Returns:
    Image (numpy.ndarray): RGB image with channels last.
    """
    image[:, :, 1] = image[:, :, 1] - 0.5
    image[:, :, 2] = image[:, :, 2] - 0.5

    r  = np.dot(image, np.array([1.0, 0.0, 1.5748]))
    g = np.dot(image, np.array([1.0, -0.1873, -0.4681]))
    b = np.dot(image, np.array([1.0, 1.8556, 0.0]))

    rgb_image = np.stack((r, g, b), axis=-1, dtype=np.float32)
    rgb_image = np.clip(rgb_image, 0.0, 1.0)

    return rgb_image

# Load your ONNX model using DirectML
models_dir = Path("E:/Code/artcnn/ONNX")
model_y = "ArtCNN_R16F96.onnx"
model_cbcr = "ArtCNN_R8F64_Chroma.onnx"

model = models_dir / model_y
session = ort.InferenceSession(model, providers=["DmlExecutionProvider"])

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print("Input name:", input_name)
print("Input shape:", input_shape)
print("Input type:", input_type)

# Load input
input = cv2.imread("./input.png", cv2.IMREAD_COLOR)
input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB, 0)
input = np.array(input).astype(np.float32) / 255.0
input = np.clip(input, 0.0, 1.0)
input = rgb2ycbcr_bt709(input)
(input_y, input_cb, input_cr) = cv2.split(input)
input_y = np.expand_dims(input_y, axis=0)
input_y = np.expand_dims(input_y, axis=0)

# Run inference
pred_y = session.run(None, {input_name: input_y})

# Print output
print("Output:", pred_y)

model = models_dir / model_cbcr
session = ort.InferenceSession(model, providers=["DmlExecutionProvider"])

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print("Input name:", input_name)
print("Input shape:", input_shape)
print("Input type:", input_type)

input_cb = cv2.resize(input_cb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR_EXACT)
input_cr = cv2.resize(input_cr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR_EXACT)
input_cb = np.expand_dims(input_cb, axis=0)
input_cr = np.expand_dims(input_cr, axis=0)
input_y = np.expand_dims(np.squeeze(pred_y), axis=0)
input_cbcr = np.concatenate((input_y, input_cb, input_cr), axis=0)
input_cbcr = np.expand_dims(input_cbcr, axis=0)

print(input_cbcr.shape)

# Run inference
pred_cbcr = session.run(None, {input_name: input_cbcr})

# Save output
pred_cbcr = np.array(pred_cbcr)
pred_cbcr = np.squeeze(pred_cbcr)
pred_cbcr = np.transpose(pred_cbcr, (1, 2, 0))
pred = np.concatenate((np.expand_dims(np.squeeze(np.array(pred_y)), axis=-1), np.squeeze(pred_cbcr)), axis=-1, dtype=np.float32)
pred = ycbcr2rgb_bt709(pred)
pred = pred * 255.0
pred = np.around(pred).astype(np.uint8)
pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR, 0)
cv2.imwrite(f"./ArtCNN_YCbCr_Combined.png", pred)
