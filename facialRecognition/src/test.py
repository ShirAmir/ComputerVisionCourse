# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import eigenfaces as ef

def load_train_data(fn="train_data.npz"):
    """ loads the trained data.
    :param fn: the loaded file name
    """

    train_data = np.load(fn)
    eigenfaces = train_data['eigenfaces']
    mean_vecs = train_data['mean_vecs']
    labels = train_data['labels_unique']
    cov_mat = train_data['cov_mat']

    return eigenfaces, mean_vecs, labels, cov_mat

def test_face(face, eigenfaces, mean_vecs, labels, cov_mat, thresh=10.0):
    """ recognizes the facae as a person in the database
    :param face: a scaled face image
    :param eigenfaces: the eigenfaces
    :param mean_vecs: the mean projection of each data set
    :param labels: the label of each person in the database
    :param cov_mat: the covariance matrix of each data set
    :param thresh: threshold of maximal considered distance
    """

    eigenvecs = eigenfaces.reshape((-1, eigenfaces.shape[-1]))
    mahal_dist = ef.compute_mahal_dist(face, eigenvecs, mean_vecs, cov_mat)
    class_ind, ratio_test = ef.classify(mahal_dist)
    min_dist = mahal_dist[class_ind]
    if min_dist > thresh:
        result_person = 'unknown'
    else:
        result_person = labels[class_ind]
    return result_person, min_dist

def image_db_testing(dir_path, thresh):
    """ test all the images in the database
    :param dir_path: path to directory with testing images.
    """

    # initialize counters
    true_positive = 0
    false_negative = 0
    false_positive = 0

    # Load tested images
    test_images, test_labels = ef.load_dataset(dir_path)
    # Load the trained data
    eigenfaces, mean_vecs, labels, cov_mat = load_train_data()

    for i, img in enumerate(test_images):
        result_person, _ = test_face(img, eigenfaces, mean_vecs, labels, cov_mat, thresh)
        if test_labels[i] in labels: # if the new face belongs to a trained person
            if result_person == test_labels[i]:
                true_positive = true_positive + 1
            else:
                print('false negative index: %s %d' % (test_labels[i], i))
                false_negative = false_negative + 1
        elif result_person != 'unknown':
            false_positive = false_positive + 1
    
    precision = true_positive / (true_positive + false_positive) * 100
    recall = true_positive / (true_positive + false_negative) * 100
    return precision, recall
    
def run_testing(img_path, output_dir, thresh):
    """ Recognizes people from the database in the image.
    :param img_path: path to the tested image
    :param output_dir: path to the output directory
    :param thresh: threshold of maximal considered distance
    """
    print("------------TEST-------------")

    # Load the trained data
    print("Load trained data.")
    eigenfaces, mean_vecs, labels, cov_mat = load_train_data()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    faces, faces_coor = ef.find_faces(img)
    label_list = []
    print("Finding faces.")
    for i in range(len(faces)):
        result_person, _ = test_face(faces[i], eigenfaces, mean_vecs, labels, cov_mat, thresh)
        label_list.append(result_person)

    img = cv2.imread(img_path)
    i = 0
    print("Marking faces.")
    for (x,y,w,h) in faces_coor:
        if label_list[i] == 'unknown':
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label_list[i], (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        i = i+1
    
    # Find a file name that isn't taken
    i = 1
    while os.path.isfile('%s%s%d%s' % (output_dir, 'result', i, '.tif')):
        i = i + 1

    # Save the result
    print("Saving results. ")
    res_path = '%s%s%d%s' % (output_dir, 'result', i, '.tif')
    cv2.imwrite(res_path, img)
    print("-----------------------------")

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return res_path

if __name__ == "__main__":
    precision_list = []
    recall_list = []
    START = 7
    END = 20
    i = START
    while i <= END:
        print(i)
        precision, recall = image_db_testing('../images/test_data_sets', i)
        precision_list.append(precision)
        recall_list.append(recall)
        i = i + 1

    # Plot results
    plt.subplot(121)
    plt.plot(range(START, END+1), precision_list, 'r--')
    plt.ylabel('precision [in %]')
    plt.xlabel('threshold')
    plt.title('precision(threshold)')
    plt.subplot(122)
    plt.plot(range(START, END+1), recall_list, 'r--')
    plt.ylabel('recall [in %]')
    plt.xlabel('threshold')
    plt.title('recall(threshold)')    
    plt.show()
   
