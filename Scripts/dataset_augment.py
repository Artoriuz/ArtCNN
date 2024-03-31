from wand.image import Image
from pathlib import Path
from tqdm import tqdm
import glob
import os

rotations = [0, 90, 180, 270]

filelist = sorted(glob.glob('./*.png'))
for myFile in tqdm(filelist):
    with Image(filename = myFile) as image:
        for rotation in rotations:
            temp = image.clone()
            temp.rotate(rotation, 'red', True)
            temp.save(filename = Path(myFile).stem + str(rotation) + ".png")
        for rotation in rotations:
            temp = image.clone()
            temp.flop()
            temp.rotate(rotation, 'red', True)
            temp.save(filename = Path(myFile).stem + str(rotation) + "f.png")
    os.remove(myFile)