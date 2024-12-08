from pathlib import Path
from tqdm import tqdm
import cv2
import glob
import os

filelist = sorted(glob.glob('./*.png'))
for myFile in tqdm(filelist):
    image = cv2.imread(myFile, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    even = image[::2, :]
    odd = image[1::2, :]
    cv2.imwrite(f"even_{Path(myFile).stem}.png", even)
    cv2.imwrite(f"odd_{Path(myFile).stem}.png", odd)
    os.remove(myFile)
