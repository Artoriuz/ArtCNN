import cv2
import numpy as np

input = cv2.imread('input.png', cv2.IMREAD_COLOR)

input = cv2.cvtColor(input, cv2.COLOR_BGR2YCrCb, 0)
(input_y, input_cr, input_cb) = cv2.split(input)
input[:,:,0] = input_y
input_cr = cv2.resize(input_cr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
input_cr = cv2.resize(input_cr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
input[:,:,1] = input_cr
input_cb = cv2.resize(input_cb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
input_cb = cv2.resize(input_cb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
input[:,:,2] = input_cb
input = cv2.cvtColor(input, cv2.COLOR_YCrCb2BGR, 0)
cv2.imwrite('input_420.png', input)
