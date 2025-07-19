import onnxruntime as ort
import numpy as np


class Engine:
    def __init__(self, model, providers=None):
        if providers:
            self.providers = providers
        else:
            # self.providers = ["MIGraphXExecutionProvider", "TensorrtExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
            self.providers = ["DmlExecutionProvider"]
        self.model = model
        self.session = ort.InferenceSession(self.model, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type

    def fix_input_shape(self, input):
        if input.ndim == 2:
            input = np.expand_dims(input, axis=0)
            input = np.expand_dims(input, axis=0)
        elif input.ndim == 3:
            input = np.expand_dims(input, axis=0)

        if input.shape[3] == 1 or input.shape[3] == 3:
            input = input.transpose(0, 3, 1, 2)
        return input

    def fix_output_shape(self, output):
        output = np.squeeze(np.array(output))
        if output.ndim == 2:
            output = np.expand_dims(output, axis=-1)
        elif output.ndim == 3:
            output = output.transpose(1, 2, 0)
        return output

    def run(self, input):
        input = self.fix_input_shape(input)
        pred = self.session.run(None, {self.input_name: input})
        pred = self.fix_output_shape(pred)
        return pred
