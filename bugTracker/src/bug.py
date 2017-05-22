# *************************************************
# ********** Multi Object Motion Tracker **********
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import numpy.random as rnd
import cv2

STATE_NUM = 4
MEASURE_NUM = 2


class Bug:
    bug_counter = 1
    max_path_length = 20

    def __init__(self, start_point):
        # The bug's kalman tracker
        self.kalman = cv2.KalmanFilter(STATE_NUM, MEASURE_NUM)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.3
        # The bug's unique identifier
        self.id = Bug.bug_counter
        Bug.bug_counter = Bug.bug_counter + 1

        # The bug's associated color
        self.color = (rnd.randint(255), rnd.randint(255), rnd.randint(255))

        # The bug's path
        self.path = [start_point]

    def update_path(self, path_point):
        if len(self.path) == Bug.max_path_length:
            del self.path[0]
        self.path.append(path_point)

    def __str__(self):
        bug_str = 'bug ' + str(self.id)
        return bug_str
