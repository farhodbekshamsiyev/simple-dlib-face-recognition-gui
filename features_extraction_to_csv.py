"""
F. Shamsiyev
"""
# Extract features from images and save into "features_all.csv"

import os
import dlib
from skimage import io
import csv
import numpy as np

# Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

# Dlib Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1(
    "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# Return 128D features for single image
# Input:    path_img           <class 'str'>
# Output:   face_descriptor    <class 'dlib.vector'>
def return_128d_features(path_img):
    img_rd = io.imread(path_img)
    faces = detector(img_rd, 1)

    print("%-40s %-20s" % (" >> Image with faces detected:", path_img), '\n')

    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        print("no face")
    return face_descriptor


# Return the mean value of 128D face descriptor for person X
# Input:    path_faces_personX       <class 'str'>
# Output:   features_mean_personX    <class 'numpy.ndarray'>
def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for i in range(len(photos_list)):
            # Get 128D features for single image of personX
            print("%-40s %-20s" % (" >> Reading image:", path_faces_personX + "/" + photos_list[i]))
            features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
            # Jump if no face detected from image
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        print(" >> Warning: No images in " + path_faces_personX + '/', '\n')

    # Compute the mean
    # personX N x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=int, order='C')
    return features_mean_personX


def get_feature(names):
    # Get the order of latest person
    person_list = os.listdir("data/data_faces_from_camera/")
    person_num_list = []
    for person in person_list:
        person_num_list.append(int(person.split('_')[-1]))
    person_cnt = max(person_num_list)

    with open("data/features_all.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in range(person_cnt):
            # Get the mean/average features of face/personX, it will be a list with a length of 128D
            print(path_images_from_camera + names[person])
            features_mean_personX = return_features_mean_personX(
                path_images_from_camera + names[person])
            writer.writerow(features_mean_personX)
            print(" >> The mean of features:", list(features_mean_personX), '\n')
        print("Save all the features of faces registered into: data/features_all.csv")
