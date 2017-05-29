# *************************************************
# ********** Multi Object Motion Tracker **********
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import numpy.random as rnd
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import bug
import os

def track(video_path, contour_size_thresh, output_dir):
    FRAME_HISTORY = 10
    EXIT_BORDER = 10

    # Initialize parameters
    video = cv2.VideoCapture(video_path)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=FRAME_HISTORY, nmixtures=2,
                                                    backgroundRatio=0.1, noiseSigma=0)
    bugs = []
    frame_num = 0
    show_box = 1
    show_trail = 1

    # Find a file name that isn't taken
    i = 1
    while os.path.isfile('%s%s%d%s' % (output_dir, '/result', i, '.mp4')):
        i = i + 1 

    # Capture frame-by-frame
    ret, frame = video.read()

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    frame_size = (np.shape(frame)[1], np.shape(frame)[0])
    new_video = cv2.VideoWriter('%s%s%d%s' % (output_dir, '/result', i, '.mp4'), fourcc, 20.0, frame_size)

    while (True):
        # Capture frame-by-frame
        ret, frame = video.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        disp_frame = frame.copy()
        frame_num = frame_num + 1
        # cv2.putText(frame, "#%d" % frame_num, (3, 12), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 127, 0), 1)

        print("Frame #%d | " % frame_num, end="")

        # Apply Background Subtraction
        fgmask = fgbg.apply(frame)

        if frame_num < FRAME_HISTORY * 2.5:
            continue

        # Perform morphological transformations
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilation = cv2.dilate(fgmask, kernel_ellipse, iterations=1)
        fgmask = cv2.erode(dilation, kernel_ellipse, iterations=1)
        #cv2.imshow('bg', fgmask)

        # Find object contours
        _, contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours[:] = [item for i, item in enumerate(contours) if cv2.contourArea(contours[i]) >= contour_size_thresh]

        print("contours found: %d" % len(contours))

        centroids_list = []
        for curr_contour in contours:
            # Find contour's centriod
            moment = cv2.moments(curr_contour)
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            measure = np.array([cx, cy], dtype=np.float32)
            centroids_list.append(measure)
            # for debug -- show centriod
            # TODO: move drawing to final drawing stage
            cv2.circle(disp_frame, (cx, cy), 3, (255, 0, 0), 1)
        centroids = np.array(centroids_list)

        if len(bugs) == 0:
            # Init the bug list
            for cent in centroids:
                buggy = bug.Bug(cent)
                bugs.append(buggy)
        else:
            if len(centroids):
                cost_matrix = np.zeros((len(bugs), len(centroids)))

                for i, buggy in enumerate(bugs):
                    last_pos = buggy.path[-1]
                    cost_matrix[i, :] = np.sqrt((centroids[:, 0] - last_pos[0]) ** 2 +
                                                (centroids[:, 1] - last_pos[1]) ** 2)

                # Hungarian Algorithm Assignment
                bug_ind, centroid_ind = linear_sum_assignment(cost_matrix)

                for i in range(len(centroid_ind)):
                    bugs[bug_ind[i]].update_path(centroids[centroid_ind[i]])
                    bugs[bug_ind[i]].plot_on_img(disp_frame, show_box, show_trail, contours[centroid_ind[i]])

                # Number of bugs < number of centroids
                # We want to add more bugs
                if len(bugs) < len(centroids):
                    new_centrois_inds = [x for x in range(len(centroids)) if x not in centroid_ind]
                    for c_ind in new_centrois_inds:
                        bugs.append(bug.Bug(centroids[c_ind]))

                # Number of begs < number of centroids
                # We want to check if bugs merged or bugs left
                elif len(bugs) > len(centroids):
                    # First check if we need to delete bugs
                    unassigned_bugs = [(i, x) for i, x in enumerate(bugs) if i not in bug_ind]
                    bugs_to_delete = []

                    for b_ind, b in unassigned_bugs:
                        x, y = b.path[-1]
                        if ((x < EXIT_BORDER) or (x > (frame.shape[1] - EXIT_BORDER)) or (y < EXIT_BORDER) or
                                (y > (frame.shape[0] - EXIT_BORDER))):
                            bugs_to_delete.append(b_ind)
                    bugs = [x for i, x in enumerate(bugs) if i not in bugs_to_delete]
                    print("Deleted %s" % str(bugs_to_delete))

                    # Check if bugs merged
                    unassigned_bugs = [(i, x) for i, x in enumerate(bugs) if i not in bug_ind]
                    for b_ind, b in unassigned_bugs:
                        last_pos = b.path[-1]
                        distance = np.sqrt((centroids[:, 0] - last_pos[0]) ** 2 + (centroids[:, 1] - last_pos[1]) ** 2)
                        closest_centroid_ind = np.argmin(distance)
                        bugs[b_ind].update_path(centroids[closest_centroid_ind])
                        bugs[b_ind].plot_on_img(disp_frame, show_box, show_trail, contours[closest_centroid_ind])

            cv2.putText(disp_frame, "%d Bugs in frame" % len(centroid_ind), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))
            # Display the resulting frame
            cv2.imshow('frame', cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB))
            new_video.write(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(60) & 0xFF
            if key == ord('t'):
                show_trail = 1 - show_trail
            elif key == ord('b'):
                show_box = 1 - show_box
            elif key == ord(' '):
                key = '0'
                while key != ord(' ') and key != 27:
                    if key == 27:
                        break
                    elif key == ord('t'):
                        show_trail = 1 - show_trail
                    elif key == ord('b'):
                        show_box = 1 - show_box
                    disp_frame = frame.copy()
                    for i in range(len(centroid_ind)):
                        bugs[bug_ind[i]].plot_on_img(disp_frame, show_box, show_trail, contours[centroid_ind[i]])
                    cv2.imshow('frame', cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB))
                    key = cv2.waitKey(0)
            if key == 27: #Esc ASCII code
                break
            key = '0'

    # When everything done, release the capture
    bug.Bug.bug_counter = 0
    video.release()
    new_video.release()
    cv2.destroyAllWindows()
