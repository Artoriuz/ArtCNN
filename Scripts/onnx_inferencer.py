import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

# Load your ONNX model using DirectML
models_dir = Path("E:/Code/artcnn/ONNX")
models = ["ArtCNN_R8F64.onnx", "ArtCNN_R16F96.onnx"]

for model in models:
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
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY, 0)
    input = np.array(input).astype(np.float32) / 255.0
    input = np.clip(input, 0.0, 1.0)
    input = np.expand_dims(input, axis=0)
    input = np.expand_dims(input, axis=0)

    # Run inference
    pred = session.run(None, {input_name: input})

    # Print output
    print("Output:", pred)

    # Save output
    pred = np.squeeze(pred)
    pred = pred * 255.0
    pred = np.squeeze((np.around(pred)).astype(np.uint8))
    cv2.imwrite(f"./{Path(model).stem}.png", pred)
