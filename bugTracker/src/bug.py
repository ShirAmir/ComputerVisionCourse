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
    bug_counter = 0
    max_path_length = 20

    def __init__(self, start_point):
        # The bug's kalman tracker
        self.kalman = cv2.KalmanFilter(STATE_NUM, MEASURE_NUM, 0)
        # H
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)

        # F. In each state we take the previous location with the previous velocity
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]],
                                                np.float32)

        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]],
                                               np.float32) * 0.03
        # self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10
        # self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

        # initial state
        self.kalman.statePre = np.array([start_point[0], start_point[1], 0, 0], dtype=np.float32)

        # The bug's unique identifier
        self.id = Bug.bug_counter
        Bug.bug_counter = Bug.bug_counter + 1

        # The bug's associated color
        self.color = (rnd.randint(255), rnd.randint(255), rnd.randint(255))

        # The bug's path
        self.path = [start_point]

        self.penalty = 0

    def get_position(self):
        return self.path[-1]

    def update_path(self, point_observation):
        self.penalty = 0 # initialize penalty after rerecognition
        self.kalman.correct(point_observation)
        pred = self.kalman.predict()
        cx, cy = pred[0], pred[1]

        if len(self.path) == Bug.max_path_length:
            self.path.pop(0)
        # self.path.append(np.array([cx, cy]))
        self.path.append(point_observation)

    def plot_on_img(self, img, show_box, show_trail, contour=None):
        if show_trail == 1:
            for j, step in enumerate(reversed(self.path)):
                cv2.circle(img, (step[0], step[1]), max(1, int(4 - j * 0.3)), self.color, -1)
        if contour is not None and show_box == 1:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), self.color, 2)
            cv2.putText(img, self.__str__(), (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1.0, self.color, 1)

    def __str__(self):
        bug_str = 'bug ' + str(self.id)
        return bug_str

    def __repr__(self):
        return "<Bug: id=%d>" % self.id
