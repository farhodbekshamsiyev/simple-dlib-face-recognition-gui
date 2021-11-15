"""
F. Shamsiyev
"""
# Real-time face detection and recognition

import dlib
import numpy as np
import cv2
import pandas as pd
import os
import time
from PIL import Image, ImageDraw, ImageFont

# Dlib Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1(
    "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    image = None
    path_photos_from_camera = "data/data_faces_from_camera/"

    def __init__(self):
        self.feature_known_list = []  # Save the features of faces in the database
        self.name_known_list = []  # Save the name of faces in the database

        self.current_frame_face_cnt = 0  # Counter for faces in current frame
        self.current_frame_feature_list = []  # Features of faces in current frame
        self.current_frame_name_position_list = []  # Positions of faces in current frame
        self.current_frame_name_list = []  # Names of faces in current frame

        # Update FPS
        self.fps = 0
        self.frame_start_time = 0

    # "features_all.csv" Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                names = self.getUserNames()
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.feature_known_list.append(features_someone_arr)
                # self.name_known_list.append("Person_" + str(i + 1))
                self.name_known_list.append(names[i])
            print("Faces in Database：", len(self.feature_known_list))
            return 1
        else:
            print('##### Warning #####', '\n')
            print("'features_all.csv' not found!")
            print(
                "Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'",
                '\n')
            print('##### End Warning #####')
            return 0

    # Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def getUserNames(self):
        return os.listdir(self.path_photos_from_camera)

    def getFps(self):
        return str(self.fps.__round__(1))

    def getFaceCount(self):
        return str(self.current_frame_face_cnt)

    def draw_note(self, img_rd):
        font = cv2.FONT_ITALIC

        cv2.putText(img_rd, "Face Recognizer", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8,
        #             (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "Faces: " + str(self.current_frame_face_cnt), (20, 140), font, 0.8,
        #             (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd, names):
        # Write names under rectangle
        font = ImageFont.truetype("simsun.ttc", 30)
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # cv2.putText(img_rd, self.current_frame_name_list[i], self.current_frame_name_position_list[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            # draw.text(xy=self.current_frame_name_position_list[i],
            #           text=self.current_frame_name_list[i], font=font)
            draw.text(xy=self.current_frame_name_position_list[i],
                      text=self.current_frame_name_list[i], font=font)
            img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_with_name

    # Show names in chinese
    def show_chinese_name(self):
        pass
        # Default known name: person_1, person_2, person_3
        # if self.current_frame_face_cnt >= 1:
        #     self.name_known_list[0] = '张1'.encode('utf-8').decode()
        #     self.name_known_list[1] = '张2'.encode('utf-8').decode()
        #     self.name_known_list[2] = '张3'.encode('utf-8').decode()
        #     self.name_known_list[3] = '张4'.encode('utf-8').decode()
        #     self.name_known_list[4] = '张5'.encode('utf-8').decode()

    # Face detection and recognition from input video stream
    def process(self, frame):
        self.image = frame
        # 1. Get faces known from "features.all.csv"
        if self.get_face_database():
            print(">>> Frame start")
            faces = detector(self.image, 0)

            self.draw_note(self.image)
            self.current_frame_feature_list = []
            self.current_frame_face_cnt = 0
            self.current_frame_name_position_list = []
            self.current_frame_name_list = []

            # 2. Face detected in current frame
            if len(faces) != 0:
                # 3. Compute the face descriptors for faces in current frame
                for i in range(len(faces)):
                    shape = predictor(self.image, faces[i])
                    self.current_frame_feature_list.append(
                        face_reco_model.compute_face_descriptor(self.image, shape))
                # 4. Traversal all the faces in the database
                for k in range(len(faces)):
                    print(">>>>>> For face", k + 1, " in camera")
                    # Set the default names of faces with "unknown"
                    self.current_frame_name_list.append("unknown")

                    # Positions of faces captured
                    self.current_frame_name_position_list.append(tuple(
                        [faces[k].left(),
                         int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                    # 5.
                    # For every faces detected, compare the faces in the database
                    current_frame_e_distance_list = []
                    for i in range(len(self.feature_known_list)):
                        # person_X
                        if str(self.feature_known_list[i][0]) != '0.0':
                            # print("   >>> With person", str(i + 1), ", the e distance: ", end='')
                            e_distance_tmp = self.return_euclidean_distance(
                                self.current_frame_feature_list[k],
                                self.feature_known_list[i])
                            # print(e_distance_tmp)
                            current_frame_e_distance_list.append(e_distance_tmp)
                        else:
                            # person_X
                            current_frame_e_distance_list.append(999999999)
                    # 6. Find the one with minimum e distance
                    similar_person_num = current_frame_e_distance_list.index(
                        min(current_frame_e_distance_list))
                    # print("   >>> Minimum e distance with ",
                    #       self.name_known_list[similar_person_num], ": ",
                    #       min(current_frame_e_distance_list))

                    if min(current_frame_e_distance_list) < 0.4:
                        self.current_frame_name_list[k] = self.name_known_list[similar_person_num]
                        print("   >>> Face recognition result:  " + str(
                            self.name_known_list[similar_person_num]))
                    else:
                        print("   >>> Face recognition result: Unknown person")

                    # Draw rectangle
                    for kk, d in enumerate(faces):
                        cv2.rectangle(self.image, tuple([d.left(), d.top()]),
                                      tuple([d.right(), d.bottom()]),
                                      (0, 255, 255), 2)

                self.current_frame_face_cnt = len(faces)

                # 7. Modify name if needed
                # self.show_chinese_name()

                # 8. Draw name
                img_with_name = self.draw_name(self.image, self.getUserNames())

            else:
                img_with_name = self.image

            print(">>>>>> Faces in camera now:", self.current_frame_name_list)

            # cv2.imshow("camera", img_with_name)

            # 9. Update stream FPS
            self.update_fps()
            print(">>> Frame ends\n\n")
            return img_with_name

    # OpenCV process
    # def run(self):
    #     cap = cv2.VideoCapture(0)
    #     # cap = cv2.VideoCapture("video.mp4")
    #     cap.set(3, 480)  # 640x480
    #     self.process(cap)
    #
    #     cap.release()
    #     cv2.destroyAllWindows()

