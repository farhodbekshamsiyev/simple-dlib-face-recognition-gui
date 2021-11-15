"""
Sh. Maxmudova
"""
#  Face register

import dlib
import numpy as np
import os
import shutil
import time
import cv2

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()


class Face_Register:
    image = ""
    current_face_dir = ""
    d = None
    press_n_flag = 0

    def __init__(self):
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.font = cv2.FONT_ITALIC

        self.existing_faces_cnt = 0  # cnt for counting saved faces
        self.ss_cnt = 0  # cnt for screen shots
        self.current_frame_faces_cnt = 0  # cnt for counting faces in current frame

        self.save_flag = 1  # The flag to control if save
        self.press_n_flag = 0  # The flag to check if press 'n' before 's'

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

    # Make dir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save faces images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # Delete the old data of faces
    def pre_work_del_old_face_folders(self):
        # "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")

    # Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # Get the order of latest person
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_num_list.append(int(person.split('_')[-1]))
            self.existing_faces_cnt = max(person_num_list)
        # Start from person_1
        else:
            self.existing_faces_cnt = 0

    # Update FPS of video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def getFps(self):
        return str(self.fps.__round__(1))

    def getFaceCount(self):
        return str(self.current_frame_faces_cnt)

    # PutText on cv2 window
    def draw_note(self, img_rd):
        # Add some notes
        cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        # cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), self.font, 0.8,
        #             (0, 255, 0), 1,
        #             cv2.LINE_AA)
        # cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font,
        #             0.8, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "N: Create face folder", (20, 350), self.font, 0.8, (255, 255, 255), 1,
        #             cv2.LINE_AA)
        # cv2.putText(img_rd, "S: Save current face", (20, 400), self.font, 0.8, (255, 255, 255), 1,
        #             cv2.LINE_AA)
        # cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Main process of face detection and saving
    def process(self, frame):
        # 1. Create folders to save photos
        self.pre_work_mkdir()

        # 2. "/data/data_faces_from_camera" Uncomment if want to delete the saved faces and start from person_1
        # if os.path.isdir(self.path_photos_from_camera):
        #     self.pre_work_del_old_face_folders()

        # 3. "/data/data_faces_from_camera"
        self.check_existing_faces_cnt()

        self.image = frame
        faces = detector(self.image, 0)  # Use Dlib face detector

        # 5. Face detected
        if len(faces) != 0:
            # Show the ROI of faces
            for k, d in enumerate(faces):
                self.d = d
                # Compute the size of rectangle box
                height = (d.bottom() - d.top())
                width = (d.right() - d.left())
                hh = int(height / 2)
                ww = int(width / 2)

                # 6. 480x640 / If the size of ROI > 480x640
                if (d.right() + ww) > 640 or (d.bottom() + hh > 480) or (d.left() - ww < 0) or (
                        d.top() - hh < 0):
                    cv2.putText(self.image, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255),
                                1, cv2.LINE_AA)
                    color_rectangle = (0, 0, 255)
                    self.save_flag = 0
                    # if kk == ord('s'):
                    #     print("Please adjust your position")
                else:
                    color_rectangle = (255, 255, 255)
                    self.save_flag = 1

                cv2.rectangle(self.image,
                              tuple([d.left() - ww, d.top() - hh]),
                              tuple([d.right() + ww, d.bottom() + hh]),
                              color_rectangle, 2)

        self.current_frame_faces_cnt = len(faces)

        # Add note on cv2 window
        self.draw_note(self.image)

        # 11. Update FPS
        self.update_fps()
        # print(self.fps)
        return self.image

    def saveImage(self, image):
        height = (self.d.bottom() - self.d.top())
        width = (self.d.right() - self.d.left())
        hh = int(height / 2)
        ww = int(width / 2)
        # 7. Create blank image according to the size of face detected
        img_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)

        if self.save_flag:
            # 8. Press 's' to save faces into local images
            # Check if you have pressed 'n'
            if self.press_n_flag:
                self.ss_cnt += 1
                for ii in range(height * 2):
                    for jj in range(width * 2):
                        # print(f"{self.d.top() - hh + ii} : {self.d.left() - ww + jj}")
                        img_blank[ii][jj] = image[self.d.top() - hh + ii][
                            self.d.left() - ww + jj]
                cv2.imwrite(
                    self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg",
                    img_blank)
                print("Save into：",
                      str(self.current_face_dir) + "/img_face_" + str(
                          self.ss_cnt) + ".jpg")
            else:
                # print(self.press_n_flag)
                print("Please press 'N' and press 'S'")
        self.d = None

    def createFolder(self, name):
        # 4. Press 'n' to create the folders for saving faces
        self.existing_faces_cnt += 1
        # self.current_face_dir = self.path_photos_from_camera + "person_" + str(
        #     self.existing_faces_cnt)
        self.current_face_dir = self.path_photos_from_camera + name + "_" + str(
            self.existing_faces_cnt)
        os.makedirs(self.current_face_dir)
        print('\n')
        print("Create folders: ", self.current_face_dir)

        self.ss_cnt = 0  # Clear the cnt of screen shots
        self.press_n_flag = 1  # Pressed 'n' already

