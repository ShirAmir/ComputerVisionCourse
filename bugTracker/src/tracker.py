# *************************************************
# ********** Multi Object Motion Tracker **********
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import numpy.random as rnd
import cv2

VIDEO = r'../videos/ana.mp4'
CONTOUR_SIZE_THRESH = 10000
RECT_WIDTH = 30
RECT_HEIGHT = 20

# Initialize parameters
video = cv2.VideoCapture(VIDEO)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

while(True):
    # Capture frame-by-frame
    ret, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply Background Subtraction
    fgmask = fgbg.apply(frame)
    # Find object contours
    _, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) >= CONTOUR_SIZE_THRESH:
            clr = (rnd.randint(255), rnd.randint(255), rnd.randint(255))
            #cv2.drawContours(frame, contours, i, clr, 2, 8)
            # Find contour's centriod
            moment = cv2.moments(contours[i])
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            # for debug -- show centriod
            cv2.circle(frame, (cx,cy), 3, (255, 0, 0), -1)
            measure = np.array([[np.float32(cx)],[np.float32(cy)]])
            # Correct Measurements
            kalman.correct(measure)
            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            break
    # Predict and show results.
    pred = kalman.predict()
    cx , cy = pred[0], pred[1]
    pt1 = (cx + int(RECT_WIDTH / 2), cy - int(RECT_HEIGHT / 2))
    pt2 = (cx - int(RECT_WIDTH / 2), cy + int(RECT_HEIGHT / 2))
    cv2.rectangle(frame, pt1, pt2, (0,0,255), 2, 8)

    # Display the resulting frame
    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()