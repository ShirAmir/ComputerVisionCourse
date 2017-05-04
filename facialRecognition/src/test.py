# *************************************************
# ******** Facial Detection and Recognition *******
# ************ Merav Joseph 200652063 *************
# ************* Shir Amir 209712801 ***************
# *************************************************

import numpy as np
import cv2
import pandas as pd
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

def test_image(face, eigenfaces, mean_vecs, labels, cov_mat, thresh=10.0):
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

def test_folder(folder, eigenfaces, mean_vecs, labels, cov_mat, thresh=10.0):
    images, labels_test = ef.load_dataset(folder)
    labels_pred = []
    score = []
    for ind, im in enumerate(images):
        result_person, min_dist = test_image(im, eigenfaces, mean_vecs,
                                             labels, cov_mat, thresh)
        labels_pred.append(result_person)
        score.append(min_dist)
    return labels_test, labels_pred, score

def analyze_results(labels, labels_pred, score, in_db):
    df = pd.DataFrame(np.vstack((labels, labels_pred, score, in_db)).T,
                      columns=('gt', 'pred', 'score', 'in_db'))

    # Check if prediction is correct
    df['in_db'] = df['in_db'].astype(bool)
    df['correct'] = df['gt'] == df['pred']
    df.loc[~df['in_db'], 'correct'] = df.loc[~df['in_db'], 'pred'].isnull()

    # Divide to true/false samples
    df_in_db = df[df['in_db']]
    df_not_in_db = df[~df['in_db']]

    # analyze true samples
    num_samples = len(df)
    num_in_db_samples = len(df_in_db)
    num_not_in_db_samples = len(df_not_in_db)

    correct_db_classification = df_in_db['correct'].sum()
    incorrect_db_classification = num_in_db_samples - correct_db_classification

    correct_not_in_db = df_not_in_db['correct'].sum()
    incorrect_not_in_db = num_not_in_db_samples - correct_not_in_db

    print("==============================================")
    print("Total samples in DB: %d" % num_in_db_samples)
    print("  Correctly classified: %d (%.1f%%)" %
          (correct_db_classification, (correct_db_classification/float(num_in_db_samples))*100.0))
    print("  Incorrectly classified: %d (%.1f%%)" %
          (incorrect_db_classification, (incorrect_db_classification / float(num_in_db_samples)) * 100.0))

    print("\nTotal samples not in DB: %d" % num_not_in_db_samples)
    print("  Correctly classified as not in DB: %d (%.1f%%)" %
          (correct_not_in_db, (correct_not_in_db/float(num_not_in_db_samples))*100.0))
    print("  Incorrectly classified as person: %d (%.1f%%)" %
          (incorrect_not_in_db, (incorrect_not_in_db/float(num_not_in_db_samples))*100.0))
    print("==============================================")

    df.to_csv("result.csv")

def run_testing(img_path, output_dir):
    """ Recognizes people from the database in the image.
    :param img_path: path to the tested image
    :param output_dir: path to the output directory
    """

    # Load the trained data
    eigenfaces, mean_vecs, labels, cov_mat = load_train_data()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    faces, faces_coor = ef.find_faces(img)
    label_list = []
    for i in range(len(faces)):
        result_person, _ = test_image(faces[i], eigenfaces, mean_vecs, labels, cov_mat, thresh=10.0)
        label_list.append(result_person)
    
    img = cv2.imread(img_path)
    i = 0
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
    res_path = '%s%s%d%s' % (output_dir, 'result', i, '.tif')
    cv2.imwrite(res_path, img)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    labels_p, labels_pred_p, score_p = test_folder("../images/positive_test_images",
                                             eigenfaces, mean_vecs, labels, cov_mat, thresh=10.0)
    labels_n, labels_pred_n, score_n = test_folder("../images/negative_test_images",
                                             eigenfaces, mean_vecs, labels, cov_mat, thresh=10.0)
    labels = labels_p + labels_n
    labels_pred = labels_pred_p + labels_pred_n
    score = score_p + score_n
    in_db = [True]*len(labels_p) + [False]*len(labels_n)
    analyze_results(labels, labels_pred, score, in_db)
    print("")
    """
    return res_path

if __name__ == "__main__":
    run_testing()