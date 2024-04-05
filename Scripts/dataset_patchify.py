from wand.image import Image
from pathlib import Path
from tqdm import tqdm
import glob
import os

patch_size = 256

filelist = sorted(glob.glob('./*.png'))
for myFile in tqdm(filelist):
    with Image(filename = myFile) as image:
        i = 0
        for h in range(0, image.height, patch_size):
            for w in range(0, image.width, patch_size):
                w_end = w + patch_size
                h_end = h + patch_size
                if (w_end <= image.width and h_end <= image.height):
                    with image[w:w_end, h:h_end] as chunk:
                        chunk.save(filename = Path(myFile).stem + f"-{i:04d}.png")
                    i += 1
    os.remove(myFile)
