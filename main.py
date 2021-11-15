"""
F. Shamsiyev
"""
# import system module

import os
import sys

# import some PyQt5 modules
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMessageBox, QLabel
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2

from features_extraction_to_csv import get_feature
from get_image import Face_Register

# from ui_main_window import *
from rec_image import Face_Recognizer


class Ui(QtWidgets.QMainWindow):
    image = ""
    cam_open = False
    facedetect = ""
    facerecognition = ""
    fps = "0"
    facecount = "0"
    path_photos_from_camera = "data/data_faces_from_camera/"
    userNames = []

    # class constructor
    def __init__(self, parent=None):
        super(Ui, self).__init__(parent)
        uic.loadUi("get_image_from_camera.ui", self)
        # self.pb_camera.clicked.connect(self.loadImage)

        self.pb_create.setEnabled(False)
        self.pb_save.setEnabled(False)
        self.pb_process.setEnabled(False)
        self.rdb_rec.setEnabled(False)

        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)

        # set control_bt callback clicked  function
        self.pb_camera.clicked.connect(self.controlTimer)
        self.pb_create.clicked.connect(self.create_folder)
        self.pb_save.clicked.connect(self.save_image)
        self.pb_process.clicked.connect(self.process)
        # self.pb_quit.clicked.connect(self.quit)

    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        self.image = cv2.flip(image, 1)
        if self.rdb_det.isChecked():
            # print("Detection selected")
            image = self.facedetect.process(self.image)
            self.fps = self.facedetect.getFps()
            self.facecount = self.facedetect.getFaceCount()

        if self.rdb_rec.isChecked():
            # print("Recognition selected")
            image = self.facerecognition.process(self.image)
            self.fps = self.facerecognition.getFps()
            self.facecount = self.facerecognition.getFaceCount()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert image to RGB format
        # get image infos
        # height, width, channel = image.shape
        # step = channel * width
        # create QImage from image
        # qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        # self.lbl_camera.setPixmap(QPixmap.fromImage(qImg))
        image = QImage(image, image.shape[1], image.shape[0], image.strides[0],
                       QImage.Format_RGB888)
        self.lbl_camera.setPixmap(QPixmap.fromImage(image))

        self.statusBar().showMessage(f"Face: {self.facecount}  |   FPS: {self.fps}")

    def initDet(self):
        self.facedetect = Face_Register()
        self.facedetect.pre_work_del_old_face_folders()

    def initRec(self):
        self.facerecognition = Face_Recognizer()

    def create_folder(self):
        if self.cam_open:
            name = self.ln_name.text()
            self.facedetect.createFolder(name)
            self.ln_name.clear()

    def save_image(self, ):
        if self.cam_open:
            self.facedetect.saveImage(self.image)

    def process(self):
        self.userNames = os.listdir(self.path_photos_from_camera)
        # print(folders_rd)
        get_feature(self.userNames)
        self.info_message("All faces processed, you can start recognition")
        self.rdb_rec.setEnabled(True)

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.lbl_camera.setText("Press button Start")
            self.cam_open = True
            self.pb_create.setEnabled(True)
            self.pb_save.setEnabled(True)
            self.pb_process.setEnabled(True)
            self.pb_camera.setText("Stop")
            self.initDet()
            self.initRec()

        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.lbl_camera.setText("No data, please press button Start")
            self.cam_open = False
            self.pb_create.setEnabled(False)
            self.pb_save.setEnabled(False)
            self.pb_process.setEnabled(False)
            self.pb_camera.setText("Start")

            # self.label_fps = QLabel(f"FPS : 0")
            self.statusBar().showMessage(f"Face: 0  |   FPS: {0}")

    def quit(self):
        self.close()

    def info_message(self, text):
        # QMessageBox.information(self, '', f'{text}')
        msg = QMessageBox()
        msg.setWindowTitle("TUIT")
        msg.setText(text)
        msg.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui()
    ui.show()
    sys.exit(app.exec_())
