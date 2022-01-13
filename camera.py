# used these tutorials as references:
# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Changing_ColorSpaces_RGB_HSV_HLS.php
# https://www.geeksforgeeks.org/filter-color-with-opencv/
# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

# for drawing bounding boxes during video capture, this stack overflow post was very helpful:
# https://stackoverflow.com/questions/35533538/creating-bounding-box-across-an-object-in-a-video

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

    # blur and threshold res to get better edges
    blur = cv.blur(res, (12,12))
    grey = cv.cvtColor(blur, cv.COLOR_HSV2BGR)
    grey = cv.cvtColor(grey, cv.COLOR_BGR2GRAY)

    ret,thresh = cv.threshold(grey,20,255,0)

    # find contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # draw bounding boxes
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    #cv.imshow('thresh',thresh)

    # Display the resulting frame
    #cv.imshow('frame', hsv)
    if cv.waitKey(1) == ord('q'):
       break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
