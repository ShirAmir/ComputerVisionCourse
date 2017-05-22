# *************************************************
# ********** Multi Object Motion Tracker **********
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import numpy.random as rnd
import cv2
import bug

VIDEO = r'../videos/ana.mp4'
CONTOUR_SIZE_THRESH = 10000
RECT_WIDTH = 30
RECT_HEIGHT = 20

# Initialize parameters
video = cv2.VideoCapture(VIDEO)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
buggy = bug.Bug((0,0))
x,y,w,h= (0,0,0,0)

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
            # Find contour's centriod
            moment = cv2.moments(contours[i])
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
            # for debug -- show centriod
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            measure = np.array([[np.float32(cx)], [np.float32(cy)]])
            # Correct Measurements
            buggy.kalman.correct(measure)
            x,y,w,h = cv2.boundingRect(contours[i])
            break
    # Predict and show results.
    pred = buggy.kalman.predict()
    cx , cy = pred[0], pred[1]
    buggy.update_path((cx, cy))
    for point in buggy.path:
        cv2.circle(frame, point, 3, buggy.color, -1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),buggy.color,2)
    cv2.putText(frame, buggy.__str__(), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, buggy.color, 2)

    # Display the resulting frame
    cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()