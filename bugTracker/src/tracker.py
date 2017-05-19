# *************************************************
# ********** Multi Object Motion Tracker **********
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import cv2

cap = cv2.VideoCapture('../videos/bugs14.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    fgmask = fgbg.apply(frame)
    im2, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (255, 0, 0), 1, 8)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()