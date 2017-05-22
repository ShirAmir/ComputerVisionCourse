# *************************************************
# ********** Multi Object Motion Tracker **********
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import numpy.random as rnd
import cv2
from scipy.optimize import linear_sum_assignment
import bug

VIDEO = r'../videos/bugs14.mp4'
CONTOUR_SIZE_THRESH = 90


if __name__ == "__main__":
    # Initialize parameters
    video = cv2.VideoCapture(VIDEO)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    bugs = []
    for i in range(9):
        bugs.append(bug.Bug((0,0)))
    x,y,w,h= (0,0,0,0)

    while(True):
        # Capture frame-by-frame
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply Background Subtraction
        fgmask = fgbg.apply(frame)

        # Find object contours
        _, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours[:] = [ item for i,item in enumerate(contours) if cv2.contourArea(contours[i]) >= CONTOUR_SIZE_THRESH]

        if len(contours) >= 9:
            cost_matrix = np.zeros((9,9))
            centriod_list = []
            for i in range(9):
                moment = cv2.moments(contours[i])
                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])
                measure = np.array([[np.float32(cx)], [np.float32(cy)]])
                centriod_list.append(measure)
            # calculate cost matrix to match each contour to bug
            for i in range(9):
                for j in range(9):
                    moment = cv2.moments(contours[j])
                    cx = int(moment['m10']/moment['m00'])
                    cy = int(moment['m01']/moment['m00'])
                    cost_matrix[i][j] = np.sqrt((bugs[i].path[-1][0] - cx) ** 2  + (bugs[i].path[-1][1] - cy) ** 2)

            # Hungarian Algorithm Assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i in range(9):
                bugs[i].kalman.correct(centriod_list[col_ind[i]])
                # Predict and show results.
                pred = bugs[i].kalman.predict()
                cx , cy = pred[0], pred[1]
                bugs[i].update_path((cx, cy))
                for point in bugs[i].path:
                    cv2.circle(frame, point, 3, bugs[i].color, -1)
                x, y, w, h = cv2.boundingRect(contours[col_ind[i]])
                cv2.rectangle(frame,(x,y),(x+w,y+h),bugs[i].color,2)
                cv2.putText(frame, bugs[i].__str__(), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bugs[i].color, 2)

        """for i in range(len(contours)):
            if cv2.contourArea(contours[i]) >= CONTOUR_SIZE_THRESH:
                # Find contour's centriod
                moment = cv2.moments(contours[i])
                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])
                measure = np.array([[np.float32(cx)], [np.float32(cy)]])
                # Correct Measurements
                bugs[i].kalman.correct(measure)
                x,y,w,h = cv2.boundingRect(contours[i])
        """

        # Display the resulting frame
        cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    video.release()
    cv2.destroyAllWindows()