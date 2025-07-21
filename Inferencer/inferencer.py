import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from inout import load_image, save_image
from utils import rgb2ycbcr, ycbcr2rgb, bilinear, stack, unstack
from engine import Engine

parser = ArgumentParser(description="ONNX Inferencer")
parser.add_argument("input", help="Input image")
parser.add_argument("-m", "--model", help="ArtCNN Model", default="ArtCNN_R16F96.onnx")
parser.add_argument("-cm", "--chroma-model", help="ArtCNN Chroma Model", default="ArtCNN_R16F96_Chroma.onnx")
parser.add_argument("-t", "--task", help="Task to perform", default="luma")
args = parser.parse_args()

print(f"Scaling {args.input} with {args.model}. The task is {args.task}.")
input = Path(args.input) if args.input is not None else None
model = Path(args.model) if args.model is not None else None
chroma_model = Path(args.chroma_model) if args.chroma_model is not None else None

match args.task:
    case "luma":
        image = load_image(input)
        pred = Engine(model).run(image)
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png")
    case "rgb":
        image = load_image(input, grayscale=False)
        b, g, r = unstack(image)
        engine = Engine(model)
        pred_b = engine.run(b)
        pred_g = engine.run(g)
        pred_r = engine.run(r)
        pred = stack(pred_b, pred_g, pred_r)
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png", grayscale=False)
    case "ycbcr":
        image = load_image(input, grayscale=False)
        image = rgb2ycbcr(image)
        y, cb, cr = unstack(image)
        cb = bilinear(cb, 2.0)
        cr = bilinear(cr, 2.0)
        pred_y = Engine(model).run(y)
        pred = Engine(chroma_model).run(stack(pred_y, cb, cr))
        pred_cb, pred_cr = np.unstack(pred, axis=-1)
        output = stack(pred_y, pred_cb, pred_cr)
        output = ycbcr2rgb(output)
        save_image(output, f"{input.stem}_{model.stem}_{args.task}.png", grayscale=False)
    case _:
        print("Unsupported task")
