import cv2
import numpy as np
cap=cv2.VideoCapture('d.mp4')
fgbg=cv2.createBackgroundSubtractorKNN()
while True:
    _,img=cap.read();
    fgmask=fgbg.apply(img)
    screen_res = 420, 820
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow('Orignal', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Orignal', window_width, window_height)
    cv2.namedWindow('Background subtraction', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Background subtraction', window_width, window_height)
    cv2.imshow('Orignal', img)
    cv2.imshow('Background subtraction',fgmask)
    if cv2.waitKey(1)==27:
        break;
cap.release()
cv2.destroyWindow('0')