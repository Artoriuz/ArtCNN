import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from inout import load_image, save_image
from utils import rgb2ycbcr, ycbcr2rgb, bilinear, stack, unstack, self_ensemble_augment, self_ensemble_combine
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
        engine = Engine(model)
        image = load_image(input)
        pred = engine.run(image)
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png")

    case "rgb":
        engine = Engine(model)
        image = load_image(input, grayscale=False)
        r, g, b = unstack(image)
        pred = stack(engine.run(r), engine.run(g), engine.run(b))
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png", grayscale=False)

    case "ycbcr":
        luma_engine = Engine(model)
        chroma_engine = Engine(chroma_model)
        image = load_image(input, grayscale=False)
        image = rgb2ycbcr(image)
        y, cb, cr = unstack(image)
        cb = bilinear(cb, 2.0)
        cr = bilinear(cr, 2.0)
        pred_y = luma_engine.run(y)
        pred = chroma_engine.run(stack(pred_y, cb, cr))
        pred_cb, pred_cr = np.unstack(pred, axis=-1)
        pred = stack(pred_y, pred_cb, pred_cr)
        pred = ycbcr2rgb(pred)
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png", grayscale=False)

    case "luma-se":
        engine = Engine(model)
        image = load_image(input)
        images = self_ensemble_augment(image)
        preds = []
        for image in images:
            pred = engine.run(image)
            preds.append(pred)
        pred = self_ensemble_combine(preds)
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png")

    case "rgb-se":
        engine = Engine(model)
        image = load_image(input, grayscale=False)
        images = self_ensemble_augment(image)
        preds = []
        for image in images:
            r, g, b = unstack(image)
            pred = stack(engine.run(r), engine.run(g), engine.run(b))
            preds.append(pred)
        pred = self_ensemble_combine(preds)
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png", grayscale=False)

    case "ycbcr-se":
        luma_engine = Engine(model)
        chroma_engine = Engine(chroma_model)
        image = load_image(input, grayscale=False)
        images = self_ensemble_augment(image)
        preds = []
        for image in images:
            image = rgb2ycbcr(image)
            y, cb, cr = unstack(image)
            pred_y = luma_engine.run(y)
            cb = bilinear(cb, 2.0)
            cr = bilinear(cr, 2.0)
            pred = chroma_engine.run(stack(pred_y, cb, cr))
            pred_cb, pred_cr = np.unstack(pred, axis=-1)
            pred = stack(pred_y, pred_cb, pred_cr)
            pred = ycbcr2rgb(pred)
            preds.append(pred)
        pred = self_ensemble_combine(preds)
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png", grayscale=False)

    case "denoise-rgb":
        engine = Engine(model)
        image = load_image(input, grayscale=False)
        pred = engine.run(image)
        save_image(pred, f"{input.stem}_{model.stem}_{args.task}.png", grayscale=False)
    case _:
        print("Unsupported task")
