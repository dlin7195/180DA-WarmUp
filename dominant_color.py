# followed the tutorial:
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

# merged the histogram code with my code from camera.py
# (mainly the video capture part)

# also adjusted the output image of the bar from rgb back to bgr
# bc I changed the original use of plt.imshow to cv.imshow (which is bgr default)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

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

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    frame = frame.reshape((frame.shape[0] * frame.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(frame)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    #plt.axis("off")
    bar = cv.cvtColor(bar, cv.COLOR_RGB2BGR)
    cv.imshow('bar',bar)
    #cv.imshow('frame',frame)
    #plt.show()

    # Display the resulting frame
    #cv.imshow('frame', hsv)
    if cv.waitKey(1) == ord('q'):
       break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

