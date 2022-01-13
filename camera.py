# used these tutorials as references:
# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Changing_ColorSpaces_RGB_HSV_HLS.php
# https://www.geeksforgeeks.org/filter-color-with-opencv/
# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # threshold for blue
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # alt filter (from geeksforgeeks)
    #lower_blue = np.array([60,25,140])
    #upper_blue = np.array([180,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)

    #contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cnt = contours[0]
    #cv.drawContours(res, [cnt], 0, (0,255,0), 3)
    # couldn't figure out this part yet --> in progress

    # bounding box on res
    #x,y,w,h = cv.boundingRect(cnt)
    #cv.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)

    # Display the resulting frame
    #cv.imshow('frame', hsv)
    if cv.waitKey(1) == ord('q'):
       break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
