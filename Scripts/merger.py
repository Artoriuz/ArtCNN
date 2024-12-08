import numpy as np
import cv2
import glob

filelist = sorted(glob.glob('./*.png'))
print(filelist)
merge_in = []
for myFile in filelist:
    image = cv2.imread(myFile, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 0)
    merge_in.append(image)
merge_in = np.array(merge_in).astype(np.float32)

output = np.mean(merge_in, axis=0, keepdims=False)
output = np.round(output).astype(np.uint8)
cv2.imwrite('./merge.png', output)
