# *************************************************
# ********** Multi Object Motion Tracker **********
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import numpy.random as rnd
import cv2

VIDEO = r'../videos/bug00.mp4'
CONTOUR_SIZE_THRESH = 150

cap = cv2.VideoCapture(VIDEO)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    fgmask = fgbg.apply(frame)
    _, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) >= CONTOUR_SIZE_THRESH:
            clr = (rnd.randint(255), rnd.randint(255), rnd.randint(255))
            cv2.drawContours(frame, contours, i, clr, 2, 8)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()