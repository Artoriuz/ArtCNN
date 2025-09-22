from argparse import ArgumentParser
from pathlib import Path

from inout import load_image, save_image
from utils import rgb2ycbcr, ycbcr2rgb, bilinear, stack, unstack, self_ensemble_augment, self_ensemble_combine
from engine import Engine


def process_image(image, task, model, chroma_model=None):
    """
    Runs the appropriate inference pipeline for a single image.
    """
    if task == "luma":
        engine = Engine(model)
        return engine.run(image)

    elif task == "rgb":
        engine = Engine(model)
        r, g, b = unstack(image)
        return stack(engine.run(r), engine.run(g), engine.run(b))

    elif task == "ycbcr":
        if chroma_model is None:
            raise ValueError("Chroma model is required for YCbCr task.")
        luma_engine = Engine(model)
        chroma_engine = Engine(chroma_model)
        image = rgb2ycbcr(image)
        y, cb, cr = unstack(image)
        cb = bilinear(cb, 2.0)
        cr = bilinear(cr, 2.0)
        pred_y = luma_engine.run(y)
        pred = chroma_engine.run(stack(pred_y, cb, cr))
        pred_cb, pred_cr = unstack(pred)
        return ycbcr2rgb(stack(pred_y, pred_cb, pred_cr))

    elif task == "denoise":
        engine = Engine(model)
        return engine.run(image)

    else:
        raise ValueError(f"Unsupported task: {task}")


def run_inference(inputs, task, model, chroma_model=None, self_ensemble=False, grayscale=True):
    for input_path in inputs:
        image = load_image(input_path, grayscale=grayscale)

        if self_ensemble:
            images = self_ensemble_augment(image)
            preds = [process_image(img, task, model, chroma_model) for img in images]
            pred = self_ensemble_combine(preds)
        else:
            pred = process_image(image, task, model, chroma_model)

        save_image(pred, f"{input_path.stem}_{Path(model).stem}_{task}.png", grayscale=grayscale)


parser = ArgumentParser(description="ONNX Inferencer")
parser.add_argument("input", help="Input image or directory")
parser.add_argument("-m", "--model", help="ArtCNN Model", default="ArtCNN_R16F96.onnx")
parser.add_argument("-cm", "--chroma-model", help="ArtCNN Chroma Model")
parser.add_argument("-t", "--task", help="Task to perform", choices=["luma", "rgb", "ycbcr", "denoise"], default="luma")
parser.add_argument("--self-ensemble", action="store_true", help="Enable self-ensemble inference")
args = parser.parse_args()

input_path = Path(args.input)

if input_path.is_dir():
    inputs = sorted(input_path.glob("*"))
elif input_path.is_file():
    inputs = [input_path]
else:
    raise FileNotFoundError(f"Input path {input_path} does not exist or is not a file/directory.")

grayscale = True if args.task == "luma" else False
run_inference(inputs, args.task, args.model, args.chroma_model, args.self_ensemble, grayscale)
