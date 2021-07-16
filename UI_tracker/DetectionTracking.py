# -*- coding: utf-8 -*-
# 本程序用于视频中车辆行人等多目标检测跟踪
# @Time    : 2021/6
# @Author  : yangyang
# @Email   : yy2289150297@163.com
# @Software: PyCharm

import os
import time
from os import getcwd
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox,QDesktopWidget
from PyQt5.QtCore import *
import numpy as np
import image1
from Detector_Model import Detector
from sort import KalmanBoxTracker
from collections import deque


class Ui_MainWindow(object):
    def __init__(self, MainWindow):

        self.path = getcwd()
        self.video_path = getcwd()
        self.model_path = "../yolo-obj/yolov4-tiny.weights"  # 模型默认路径
        self.timer_camera = QtCore.QTimer()  # 定时器
        self.timer_video = QtCore.QTimer()  # 视频定时器
        self.flag_timer = ""  # 用于标记正在进行的功能项（视频/摄像）

        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)
        self.slot_init()  # 槽函数设置

        self.radioButton_countting.setChecked(True)
        self.CAM_NUM = 0  # 摄像头标号
        self.cap = cv2.VideoCapture(self.CAM_NUM)  # 屏幕画面对象
        self.cap_video = None  # 视频流对象

        self.current_image = None  # 保存的画面
        self.detected_image = None
        # 画面的检测结果
        self.dets = []
        self.boxes = []
        self.indexIDs = []
        self.cls_IDs = []

        model_path = "../yolo-obj"  # 模型文件的目录
        labelsPath = os.path.sep.join([model_path, "coco.names"])# coco.names里面的类别
        self.LABELS = open(labelsPath).read().strip().split("\n")# 返回一个列表 里面就['person','bicycle','carmotor','bike','aero','plane','bus',....]

        # 初始化用于标记框的颜色
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

        # 用于展示目标移动路径
        self.pts = [deque(maxlen=30) for _ in range(9999)]
        self.detector_model = Detector(self.model_path) #Detector 是Detector_Model的模型


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(1728, 972)
        MainWindow.setWindowFlags(Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        MainWindow.setMinimumSize(QtCore.QSize(1728, 972))
        MainWindow.setMaximumSize(QtCore.QSize(1728, 972))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/track.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolTip("")
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("#MainWindow{border-image: url(:/newPrefix/images_test/bejing.jpg);}\n"
                                 "\n"
                                 "#QInputDialog{border-image: url(:/newPrefix/images_test/light.png);}\n"
                                 "\n"
                                 "QLabel{border:5px;}\n"
                                 "QLabel::hover {\n"
                                 "border:0px;}\n"
                                 "\n"
                                 "QMenuBar{border-color:transparent;}\n"
                                 "QToolButton[objectName=pushButton_doIt]{\n"
                                 "border:5px;}\n"
                                 "\n"
                                 "QToolButton[objectName=pushButton_doIt]:hover {\n"
                                 "image:url(:/newPrefix/images_test/run_hover.png);}\n"
                                 "\n"
                                 "QToolButton[objectName=pushButton_doIt]:pressed {\n"
                                 "image:url(:/newPrefix/images_test/run_pressed.png);}\n"
                                 "\n"
                                 "QScrollBar:vertical{\n"
                                 "background:transparent;\n"
                                 "padding:2px;\n"
                                 "border-radius:4px;\n"
                                 "max-width:8px;}\n"
                                 "\n"
                                 "QScrollBar::handle:vertical{\n"
                                 "background:#9acd32;\n"
                                 "min-height:8px;\n"
                                 "border-radius:4px;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::handle:vertical:hover{\n"
                                 "background:#9eb764;}\n"
                                 "\n"
                                 "QScrollBar::handle:vertical:pressed{\n"
                                 "background:#9eb764;\n"
                                 "}\n"
                                 "QScrollBar::add-page:vertical{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "                               \n"
                                 "QScrollBar::sub-page:vertical{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::add-line:vertical{\n"
                                 "background:none;}\n"
                                 "                                 \n"
                                 "QScrollBar::sub-line:vertical{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "QScrollArea{\n"
                                 "border:0px;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar:horizontal{\n"
                                 "background:transparent;\n"
                                 "padding:0px;\n"
                                 "border-radius:4px;\n"
                                 "max-height:6px;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::handle:horizontal{\n"
                                 "background:#9acd32;\n"
                                 "min-width:8px;\n"
                                 "border-radius:4px;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::handle:horizontal:hover{\n"
                                 "background:#9eb764;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::handle:horizontal:pressed{\n"
                                 "background:#9eb764;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::add-page:horizontal{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::sub-page:horizontal{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "QScrollBar::add-line:horizontal{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "\n"
                                 "QScrollBar::sub-line:horizontal{\n"
                                 "background:none;\n"
                                 "}\n"
                                 "QToolButton::hover{\n"
                                 "border:0px;\n"
                                 "} ")
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_author = QtWidgets.QLabel(self.centralwidget)
        self.label_author.setGeometry(QtCore.QRect(810, 120, 141, 30))
        self.label_author.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(18)
        self.label_author.setFont(font)
        self.label_author.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_author.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.label_author.setObjectName("label_author")    #作者标签
        self.label_useTime = QtWidgets.QLabel(self.centralwidget)
        self.label_useTime.setGeometry(QtCore.QRect(1530, 384, 91, 51))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(18)
        self.label_useTime.setFont(font)
        self.label_useTime.setObjectName("label_useTime") # 用时标签
        self.label_class = QtWidgets.QLabel(self.centralwidget)
        self.label_class.setGeometry(QtCore.QRect(1535, 630, 86, 31))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(18)
        self.label_class.setFont(font)
        self.label_class.setObjectName("label_class")    # 类别标签
        self.label_picTime = QtWidgets.QLabel(self.centralwidget)
        self.label_picTime.setGeometry(QtCore.QRect(1480, 384, 38, 38))
        self.label_picTime.setStyleSheet("border-image: url(:/newPrefix/images_test/yongshi1.png);")
        self.label_picTime.setText("")
        self.label_picTime.setObjectName("label_picTime")   # 用时的图标设置
        self.label_picResult = QtWidgets.QLabel(self.centralwidget)
        self.label_picResult.setGeometry(QtCore.QRect(1480, 620, 41, 41))
        self.label_picResult.setStyleSheet("border-image: url(:/newPrefix/images_test/leibie1.png);")
        self.label_picResult.setText("")
        self.label_picResult.setObjectName("label_picResult")  #类别的图标设置
        self.label_display = QtWidgets.QLabel(self.centralwidget)
        self.label_display.setGeometry(QtCore.QRect(300, 270, 1152, 648))
        self.label_display.setMinimumSize(QtCore.QSize(1152, 648))
        self.label_display.setMaximumSize(QtCore.QSize(1152, 648))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(16)
        self.label_display.setFont(font)
        self.label_display.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/yyyyy.png);")
        self.label_display.setAlignment(QtCore.Qt.AlignCenter)
        self.label_display.setObjectName("label_display")  #播放标签
        self.textEdit_model = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_model.setGeometry(QtCore.QRect(50, 277, 240, 30))
        self.textEdit_model.setMinimumSize(QtCore.QSize(240, 30))
        self.textEdit_model.setMaximumSize(QtCore.QSize(240, 30))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(12)
        self.textEdit_model.setFont(font)
        self.textEdit_model.setStyleSheet("background-color: transparent;\n"
                                          "border-color: rgb(255, 0, 0);\n"
                                          "color: rgb(255,0,0);")
        self.textEdit_model.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textEdit_model.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_model.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textEdit_model.setReadOnly(True)
        self.textEdit_model.setObjectName("textEdit_model")  #模型标签
        self.toolButton_file = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_file.setGeometry(QtCore.QRect(0, 412, 50, 40))
        self.toolButton_file.setMinimumSize(QtCore.QSize(50, 39))
        self.toolButton_file.setMaximumSize(QtCore.QSize(50, 40))
        self.toolButton_file.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_file.setAutoFillBackground(False)
        self.toolButton_file.setStyleSheet("background-color: transparent;")
        self.toolButton_file.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/12.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_file.setIcon(icon1)
        self.toolButton_file.setIconSize(QtCore.QSize(50, 40))
        self.toolButton_file.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_file.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_file.setAutoRaise(False)
        # self.toolButton_file.setAutoExclusive(True)
        # self.toolButton_file.setCheckable(True)
        # self.toolButton_file.setChecked(True)
        self.toolButton_file.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_file.setObjectName("toolButton_file")  #文件按钮
        self.textEdit_camera = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_camera.setGeometry(QtCore.QRect(50, 347, 240, 30))
        self.textEdit_camera.setMinimumSize(QtCore.QSize(240, 30))
        self.textEdit_camera.setMaximumSize(QtCore.QSize(240, 30))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(12)
        self.textEdit_camera.setFont(font)
        self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 0, 0);\n"
                                           "color:rgb(255,0,0);")
        self.textEdit_camera.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textEdit_camera.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_camera.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textEdit_camera.setReadOnly(True)
        self.textEdit_camera.setObjectName("textEdit_camera") #摄像机按钮
        self.textEdit_pic = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_pic.setGeometry(QtCore.QRect(50, 417, 240, 30))
        self.textEdit_pic.setMinimumSize(QtCore.QSize(240, 30))
        self.textEdit_pic.setMaximumSize(QtCore.QSize(240, 30))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(12)
        self.textEdit_pic.setFont(font)
        self.textEdit_pic.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                        "border-color: rgb(0, 0, 0);\n"
                                        "color: rgb(255,0,0);")
        self.textEdit_pic.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textEdit_pic.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_pic.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textEdit_pic.setReadOnly(True)
        self.textEdit_pic.setObjectName("textEdit_pic") #照片标签
        self.toolButton_camera = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_camera.setGeometry(QtCore.QRect(0, 337, 50, 45))
        self.toolButton_camera.setMinimumSize(QtCore.QSize(50, 39))
        self.toolButton_camera.setMaximumSize(QtCore.QSize(50, 45))
        self.toolButton_camera.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_camera.setAutoFillBackground(False)
        self.toolButton_camera.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 0, 0);\n"
                                             "color:rgb(0, 0, 0);")
        self.toolButton_camera.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/shexiangji2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_camera.setIcon(icon2)
        self.toolButton_camera.setIconSize(QtCore.QSize(50, 39))
        self.toolButton_camera.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_camera.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_camera.setAutoRaise(False)
        self.toolButton_camera.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_camera.setObjectName("toolButton_camera")   #相机按钮
        self.toolButton_model = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_model.setGeometry(QtCore.QRect(0, 267, 50, 40))
        self.toolButton_model.setMinimumSize(QtCore.QSize(0, 0))
        self.toolButton_model.setMaximumSize(QtCore.QSize(50, 40))
        self.toolButton_model.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_model.setAutoFillBackground(False)
        self.toolButton_model.setStyleSheet("background-color: transparent;")
        self.toolButton_model.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/baocunmoxing.png"), QtGui.QIcon.Normal,
                        QtGui.QIcon.Off)
        self.toolButton_model.setIcon(icon3)
        self.toolButton_model.setIconSize(QtCore.QSize(50, 40))
        self.toolButton_model.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_model.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_model.setAutoRaise(False)
        self.toolButton_model.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_model.setObjectName("toolButton_model")  #模型按钮
        self.label_time_result = QtWidgets.QLabel(self.centralwidget)
        self.label_time_result.setGeometry(QtCore.QRect(1610, 390, 90, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_time_result.setFont(font)
        self.label_time_result.setStyleSheet("color: rgb(210,105,30);")
        self.label_time_result.setObjectName("label_time_result") # 时间结果
        self.label_class_result = QtWidgets.QLabel(self.centralwidget)
        self.label_class_result.setGeometry(QtCore.QRect(1614, 628, 215, 29))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_class_result.setFont(font)
        self.label_class_result.setStyleSheet("color: rgb(210,105,30);")
        self.label_class_result.setObjectName("label_class_result") #类别结果
        self.label_pic_detection = QtWidgets.QLabel(self.centralwidget)
        self.label_pic_detection.setGeometry(QtCore.QRect(190, 711, 46, 51))
        self.label_pic_detection.setStyleSheet("border-image: url(:/newPrefix/images_test/xingren2.png);")
        self.label_pic_detection.setText("")
        self.label_pic_detection.setObjectName("label_pic_detection") # 左下角目标检测的图标
        self.radioButton_detection = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_detection.setGeometry(QtCore.QRect(60, 716, 121, 41))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(14)
        self.radioButton_detection.setFont(font)
        self.radioButton_detection.setObjectName("radioButton_detection") #目标检测按钮
        self.radioButton_tracking = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_tracking.setGeometry(QtCore.QRect(60, 776, 121, 41))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(14)
        self.radioButton_tracking.setFont(font)
        self.radioButton_tracking.setObjectName("radioButton_tracking")  #目标跟踪按钮
        self.radioButton_countting = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_countting.setGeometry(QtCore.QRect(60, 836, 121, 41))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(14)
        self.radioButton_countting.setFont(font)
        self.radioButton_countting.setObjectName("radioButton_countting") #目标计数按钮
        self.label_pic_tracking = QtWidgets.QLabel(self.centralwidget)
        self.label_pic_tracking.setGeometry(QtCore.QRect(190, 776, 41, 41))
        self.label_pic_tracking.setStyleSheet("border-image: url(:/newPrefix/images_test/genzong1.png);")
        self.label_pic_tracking.setText("")
        self.label_pic_tracking.setObjectName("label_pic_tracking") #目标跟踪
        self.label_pic_coutting = QtWidgets.QLabel(self.centralwidget)
        self.label_pic_coutting.setGeometry(QtCore.QRect(190, 836, 41, 41))
        self.label_pic_coutting.setStyleSheet("border-image: url(:/newPrefix/images_test/jishuqi1.png);")
        self.label_pic_coutting.setText("")
        self.label_pic_coutting.setObjectName("label_pic_coutting")#目标计数
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 676, 291, 41))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        self.label_title.setGeometry(QtCore.QRect(560, 50, 671, 51))
        self.label_title.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(26)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_title.setFont(font)
        self.label_title.setStyleSheet("color: rgb(0, 0, 0);\n"
                                       "font: italic 26pt \"隶书\";")
        self.label_title.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.label_title.setObjectName("label_title")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(0, 886, 291, 41))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_pic = QtWidgets.QLabel(self.centralwidget)
        self.label_pic.setGeometry(QtCore.QRect(10, 556, 251, 121))
        self.label_pic.setStyleSheet("border-image: url(:/newPrefix/images_test/sort5.jpg);")
        self.label_pic.setText("")
        self.label_pic.setObjectName("label_pic")        #左侧中间的 行车图片
        self.label_picNumber = QtWidgets.QLabel(self.centralwidget)
        self.label_picNumber.setGeometry(QtCore.QRect(1480, 460, 41, 41))
        self.label_picNumber.setStyleSheet("border-image: url(:/newPrefix/images_test/shumu1.png);")
        self.label_picNumber.setText("")
        self.label_picNumber.setObjectName("label_picNumber") #目标数目图片 label_picNumber
        self.label_objNum = QtWidgets.QLabel(self.centralwidget)
        self.label_objNum.setGeometry(QtCore.QRect(1530, 470, 131, 31))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(18)
        self.label_objNum.setFont(font)
        self.label_objNum.setObjectName("label_objNum") #目标数目 label_objNum
        self.label_numer_result = QtWidgets.QLabel(self.centralwidget)
        self.label_numer_result.setGeometry(QtCore.QRect(1660, 465, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_numer_result.setFont(font)
        self.label_numer_result.setStyleSheet("color: rgb(210,105,30);")
        self.label_numer_result.setObjectName("label_numer_result") # 目标数目（数字） label_numer_result
        self.comboBox_select = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_select.setGeometry(QtCore.QRect(1530, 560, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.comboBox_select.setFont(font)
        self.comboBox_select.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.comboBox_select.setStyleSheet("color: rgb(43, 89, 124);\n"
                                           "font: italic 12pt \"Consolas\";")
        self.comboBox_select.setIconSize(QtCore.QSize(36, 36))
        self.comboBox_select.setObjectName("comboBox_select") # 多选框 comboBox_select
        self.comboBox_select.addItem("")
        self.label_picSelect = QtWidgets.QLabel(self.centralwidget)
        self.label_picSelect.setGeometry(QtCore.QRect(1475, 550, 51, 51))
        self.label_picSelect.setStyleSheet("border-image: url(:/newPrefix/images_test/mubiao2.png);")
        self.label_picSelect.setText("")
        self.label_picSelect.setObjectName("label_picSelect") # 多选框的图标 label_picSelect
        self.label_conf = QtWidgets.QLabel(self.centralwidget)
        self.label_conf.setGeometry(QtCore.QRect(1535, 700, 111, 31))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(18)
        self.label_conf.setFont(font)
        self.label_conf.setObjectName("label_conf") # 置信度的方框  label_conf
        self.label_picConf = QtWidgets.QLabel(self.centralwidget)
        self.label_picConf.setGeometry(QtCore.QRect(1480, 690, 41, 41))
        self.label_picConf.setStyleSheet("border-image: url(:/newPrefix/images_test/zhixindu2.png);")
        self.label_picConf.setText("")
        self.label_picConf.setObjectName("label_picConf")  # 置信度的图标 label_picConf
        self.label_score_result = QtWidgets.QLabel(self.centralwidget)
        self.label_score_result.setGeometry(QtCore.QRect(1638, 698, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_score_result.setFont(font)
        self.label_score_result.setStyleSheet("color: rgb(210,105,30);")
        self.label_score_result.setObjectName("label_score_result")  # 置信度的结果  label_score_result
        self.label_picLocation = QtWidgets.QLabel(self.centralwidget)
        self.label_picLocation.setGeometry(QtCore.QRect(1480, 780, 41, 41))
        self.label_picLocation.setStyleSheet("border-image: url(:/newPrefix/images_test/weizhi1.png);")
        self.label_picLocation.setText("")
        self.label_picLocation.setObjectName("label_picLocation") #位置图标 label_picLocation
        self.label_location = QtWidgets.QLabel(self.centralwidget)
        self.label_location.setGeometry(QtCore.QRect(1530, 790, 131, 31))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(18)
        self.label_location.setFont(font)
        self.label_location.setObjectName("label_location")  #位置方框
        self.label_xmin = QtWidgets.QLabel(self.centralwidget)
        self.label_xmin.setGeometry(QtCore.QRect(1480, 840, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_xmin.setFont(font)
        self.label_xmin.setStyleSheet("font: italic 14pt \"Consolas\";")
        self.label_xmin.setObjectName("label_xmin") # x最小值的框
        self.label_xmax = QtWidgets.QLabel(self.centralwidget)
        self.label_xmax.setGeometry(QtCore.QRect(1480, 880, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_xmax.setFont(font)
        self.label_xmax.setStyleSheet("font: italic 14pt \"Consolas\";")
        self.label_xmax.setObjectName("label_xmax")  # x最大值的框
        self.label_ymin = QtWidgets.QLabel(self.centralwidget)
        self.label_ymin.setGeometry(QtCore.QRect(1610, 840, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_ymin.setFont(font)
        self.label_ymin.setStyleSheet("font: italic 14pt \"Consolas\";")
        self.label_ymin.setObjectName("label_ymin")  # y最小值的框
        self.label_ymax = QtWidgets.QLabel(self.centralwidget)
        self.label_ymax.setGeometry(QtCore.QRect(1610, 880, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_ymax.setFont(font)
        self.label_ymax.setStyleSheet("font: italic 14pt \"Consolas\";")
        self.label_ymax.setObjectName("label_ymax") # y最大值的框
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(1460, 740, 271, 41))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(1460, 510, 271, 41))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.label_website = QtWidgets.QLabel(self.centralwidget)
        self.label_website.setGeometry(QtCore.QRect(770, 170, 231, 31))
        self.label_website.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setFamily("Times new roman")
        font.setPointSize(18)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label_website.setFont(font)
        self.label_website.setStyleSheet("color: rgb(0, 0, 0);\n"
                                         "font: italic 18pt \"Times new roman\";")
        self.label_website.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.label_website.setObjectName("label_website")
        self.label_xmin_result = QtWidgets.QLabel(self.centralwidget)
        self.label_xmin_result.setGeometry(QtCore.QRect(1545, 840, 51, 31))
        font = QtGui.QFont()
        font.setFamily("SimSun-ExtB")
        font.setPointSize(14)
        self.label_xmin_result.setFont(font)
        self.label_xmin_result.setStyleSheet("color: rgb(210,105,31);")
        self.label_xmin_result.setObjectName("label_xmin_result")  # x最小值  的结果
        self.label_ymin_result = QtWidgets.QLabel(self.centralwidget)
        self.label_ymin_result.setGeometry(QtCore.QRect(1675, 840, 51, 31))
        font = QtGui.QFont()
        font.setFamily("SimSun-ExtB")
        font.setPointSize(14)
        self.label_ymin_result.setFont(font)
        self.label_ymin_result.setStyleSheet("color: rgb(210,105,30);")
        self.label_ymin_result.setObjectName("label_ymin_result")   # y最小值  的结果
        self.label_xmax_result = QtWidgets.QLabel(self.centralwidget)
        self.label_xmax_result.setGeometry(QtCore.QRect(1545, 880, 51, 31))
        font = QtGui.QFont()
        font.setFamily("SimSun-ExtB")
        font.setPointSize(14)
        self.label_xmax_result.setFont(font)
        self.label_xmax_result.setStyleSheet("color: rgb(210,105,30);")
        self.label_xmax_result.setObjectName("label_xmax_result")   # x最大值  的结果
        self.label_ymax_result = QtWidgets.QLabel(self.centralwidget)
        self.label_ymax_result.setGeometry(QtCore.QRect(1675, 880, 51, 31))
        font = QtGui.QFont()
        font.setFamily("SimSun-ExtB")
        font.setPointSize(14)
        self.label_ymax_result.setFont(font)
        self.label_ymax_result.setStyleSheet("color: rgb(210,105,30);")
        self.label_ymax_result.setObjectName("label_ymax_result")   # y最大值  的结果
        self.label_pic_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_pic_2.setGeometry(QtCore.QRect(1, 1, 161, 51))
        self.label_pic_2.setStyleSheet("border-image: url(:/newPrefix/images_test/huise.jpg);")
        self.label_pic_2.setText("")
        self.label_pic_2.setObjectName("label_pic_2")
        self.toolButton_saveing = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_saveing.setGeometry(QtCore.QRect(11, 11, 31, 26))
        self.toolButton_saveing.setMaximumSize(QtCore.QSize(50, 45))
        self.toolButton_saveing.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_saveing.setAutoFillBackground(False)
        self.toolButton_saveing.setStyleSheet("background-color: transparent;\n"
                                              "border-color: rgb(0, 0, 0);\n"
                                              "color:rgb(0, 0, 0);")
        self.toolButton_saveing.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/baocun2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_saveing.setIcon(icon4)
        self.toolButton_saveing.setIconSize(QtCore.QSize(50, 39))
        self.toolButton_saveing.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_saveing.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_saveing.setAutoRaise(False)
        self.toolButton_saveing.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_saveing.setObjectName("toolButton_saveing")   #保存
        self.toolButton_version = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_version.setGeometry(QtCore.QRect(118, 12, 31, 26))
        self.toolButton_version.setMaximumSize(QtCore.QSize(50, 45))
        self.toolButton_version.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_version.setAutoFillBackground(False)
        self.toolButton_version.setStyleSheet("background-color: transparent;\n"
                                              "border-color: rgb(0, 0, 0);\n"
                                              "color:rgb(0, 0, 0);")
        self.toolButton_version.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/banben2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_version.setIcon(icon5)
        self.toolButton_version.setIconSize(QtCore.QSize(50, 39))
        self.toolButton_version.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_version.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_version.setAutoRaise(False)
        self.toolButton_version.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_version.setObjectName("toolButton_version")  #版本
        self.toolButton_author = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_author.setGeometry(QtCore.QRect(81, 11, 31, 26))
        self.toolButton_author.setMaximumSize(QtCore.QSize(50, 45))
        self.toolButton_author.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_author.setAutoFillBackground(False)
        self.toolButton_author.setStyleSheet("background-color: transparent;\n"
                                             "border-color: rgb(0, 0, 0);\n"
                                             "color:rgb(0, 0, 0);")
        self.toolButton_author.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/zhuti.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_author.setIcon(icon6)
        self.toolButton_author.setIconSize(QtCore.QSize(50, 39))
        self.toolButton_author.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_author.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_author.setAutoRaise(False)
        self.toolButton_author.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_author.setObjectName("toolButton_author")  #应用主旨
        self.toolButton_settings = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_settings.setGeometry(QtCore.QRect(40, 8, 41, 31))
        self.toolButton_settings.setMaximumSize(QtCore.QSize(50, 45))
        self.toolButton_settings.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_settings.setAutoFillBackground(False)
        self.toolButton_settings.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 0, 0);\n"
                                               "color:rgb(0, 0, 0);")
        self.toolButton_settings.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/weizhi2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_settings.setIcon(icon7)
        self.toolButton_settings.setIconSize(QtCore.QSize(50, 39))
        self.toolButton_settings.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_settings.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_settings.setAutoRaise(False)
        self.toolButton_settings.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_settings.setObjectName("toolButton_settings")  #默认保存
        self.textEdit_video = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_video.setGeometry(QtCore.QRect(50, 486, 240, 30))
        self.textEdit_video.setMinimumSize(QtCore.QSize(240, 30))
        self.textEdit_video.setMaximumSize(QtCore.QSize(240, 30))
        font = QtGui.QFont()
        font.setFamily("华文仿宋")
        font.setPointSize(12)
        self.textEdit_video.setFont(font)
        self.textEdit_video.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                          "border-color: rgb(0, 0, 0);\n"
                                          "color: rgb(255,0,0);")
        self.textEdit_video.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textEdit_video.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.textEdit_video.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textEdit_video.setReadOnly(True)
        self.textEdit_video.setObjectName("textEdit_video")   #请选择视频文件
        self.toolButton_video = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_video.setGeometry(QtCore.QRect(0, 482, 50, 39))
        self.toolButton_video.setMinimumSize(QtCore.QSize(50, 39))
        self.toolButton_video.setMaximumSize(QtCore.QSize(50, 40))
        self.toolButton_video.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.toolButton_video.setAutoFillBackground(False)
        self.toolButton_video.setStyleSheet("background-color: transparent;")
        self.toolButton_video.setText("")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/shipin1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_video.setIcon(icon8)
        self.toolButton_video.setIconSize(QtCore.QSize(33, 33))
        self.toolButton_video.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.toolButton_video.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_video.setAutoRaise(False)
        self.toolButton_video.setArrowType(QtCore.Qt.NoArrow)
        self.toolButton_video.setObjectName("toolButton_video")
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionGoogle_Translate = QtWidgets.QAction(MainWindow)
        self.actionGoogle_Translate.setObjectName("actionGoogle_Translate")
        self.actionHTML_type = QtWidgets.QAction(MainWindow)
        self.actionHTML_type.setObjectName("actionHTML_type")
        self.actionsoftware_version = QtWidgets.QAction(MainWindow)
        self.actionsoftware_version.setObjectName("actionsoftware_version")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pedestrian Detection(Hmx) v1.0"))
        self.label_author.setToolTip(
            _translate("MainWindow", "<html><head/><body><p>检测跟踪机(Hmx)（邮箱：yy2289150297@163.com）</p></body></html>"))
        self.label_author.setText(_translate("MainWindow", "Hmx"))
        self.label_useTime.setText(_translate("MainWindow", "<html><head/><body><p>用时：</p></body></html>"))
        self.label_class.setText(_translate("MainWindow", "<html><head/><body><p>类别：<br/></p></body></html>"))
        self.label_display.setText(
            _translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.textEdit_model.setHtml(_translate("MainWindow",
                                               "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                               "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                               "p, li { white-space: pre-wrap; }\n"
                                               "</style></head><body style=\" font-family:\'华文仿宋\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                               "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;color: rgb(0,0,0);\"><span style=\" font-family:\'Adobe Devanagari\';\">请开始选择模型</span></p></body></html>"))
        self.textEdit_camera.setHtml(_translate("MainWindow",
                                                "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                                "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                                "p, li { white-space: pre-wrap; }\n"
                                                "</style></head><body style=\" font-family:\'华文仿宋\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                                "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;color: rgb(0,0,0);\"><span style=\" font-family:\'Adobe Devanagari\';\">实时摄像未开启</span></p></body></html>"))
        self.textEdit_pic.setHtml(_translate("MainWindow",
                                             "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                             "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                             "p, li { white-space: pre-wrap; }\n"
                                             "</style></head><body style=\" font-family:\'华文仿宋\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                             "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;color: rgb(0,0,0);\"><span style=\" font-family:\'Adobe Devanagari\';\">请选择图片文件</span></p></body></html>"))
        self.label_time_result.setText(_translate("MainWindow", "0 s"))
        self.label_class_result.setText(_translate("MainWindow", "None"))
        self.radioButton_detection.setText(_translate("MainWindow", "目标检测"))
        self.radioButton_tracking.setText(_translate("MainWindow", "目标跟踪"))
        self.radioButton_countting.setText(_translate("MainWindow", "跟踪计数"))
        self.label_title.setToolTip(
            _translate("MainWindow", "<html><head/><body><p>version: v1.0</p><p>author: Hmx</p></body></html>"))
        self.label_title.setText(_translate("MainWindow", "行人检测与跟踪系统 v1.0"))
        self.label_objNum.setText(_translate("MainWindow", "<html><head/><body><p>目标数目：<br/></p></body></html>"))
        self.label_numer_result.setText(_translate("MainWindow", "0"))
        self.comboBox_select.setCurrentText(_translate("MainWindow", "所有目标"))
        self.comboBox_select.setItemText(0, _translate("MainWindow", "所有目标"))
        self.label_conf.setText(_translate("MainWindow", "<html><head/><body><p>置信度：<br/></p></body></html>"))
        self.label_score_result.setText(_translate("MainWindow", "0"))
        self.label_location.setText(_translate("MainWindow", "<html><head/><body><p>位 置：<br/></p></body></html>"))
        self.label_xmin.setText(_translate("MainWindow", "xmin: "))
        self.label_xmax.setText(_translate("MainWindow", "xmax: "))
        self.label_ymin.setText(_translate("MainWindow", "ymin: "))
        self.label_ymax.setText(_translate("MainWindow", "ymax: "))
        self.label_website.setToolTip(_translate("MainWindow", "<html><head/><body><p>  &&</p><p> &&</p></body></html>"))
        self.label_website.setText(_translate("MainWindow", " "))
        self.label_xmin_result.setText(_translate("MainWindow", "0"))
        self.label_ymin_result.setText(_translate("MainWindow", "0"))
        self.label_xmax_result.setText(_translate("MainWindow", "0"))
        self.label_ymax_result.setText(_translate("MainWindow", "0"))
        self.textEdit_video.setHtml(_translate("MainWindow",
                                               "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                               "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                               "p, li { white-space: pre-wrap; }\n"
                                               "</style></head><body style=\" font-family:\'华文仿宋\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
                                               "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;color: rgb(0,0,0);\"><span style=\" font-family:\'Adobe Devanagari\';\">请选择视频文件</span></p></body></html>"))
        self.actionGoogle_Translate.setText(_translate("MainWindow", "Google Translate"))
        self.actionHTML_type.setText(_translate("MainWindow", "HTML type"))
        self.actionsoftware_version.setText(_translate("MainWindow", "software version"))

    def slot_init(self):  # 定义槽函数
        self.toolButton_file.clicked.connect(self.choose_file)#选择文件
        self.toolButton_model.clicked.connect(self.choose_model)#选择模型
        self.comboBox_select.currentIndexChanged.connect(self.select_obj)
        self.comboBox_select.highlighted.connect(self.pause_run)
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)#选择开启摄像机
        self.toolButton_video.clicked.connect(self.button_open_video_click)#选择开启视频
        self.toolButton_saveing.clicked.connect(self.save_file)
        self.toolButton_settings.clicked.connect(self.setting)
        self.toolButton_author.clicked.connect(self.disp_website)
        self.toolButton_version.clicked.connect(self.disp_version)
        self.timer_camera.timeout.connect(self.show_camera)  #timeout 暂停
        self.timer_video.timeout.connect(self.show_video)

    def pause_run(self):
        if self.comboBox_select.count() > 1:
            if self.flag_timer == "video":
                self.timer_video.stop()
            elif self.flag_timer == "camera":
                self.timer_camera.stop()

    def choose_model(self):
        # 选择模型按钮点击事件的槽函数
        self.timer_camera.stop()  # 停止摄像计时器
        self.timer_video.stop()  # 停止视频计时器
        if self.cap:
            self.cap.release()  # 释放画面帧
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧
        # icon1 = QtGui.QIcon()
        # icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/123456.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        # self.toolButton_model.setIcon(icon1)#点击后的 图表变换

        # 重置下拉选框
        self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
        self.comboBox_select.clear()
        self.comboBox_select.addItem('所有目标')
        self.comboBox_select.currentIndexChanged.connect(self.select_obj)
        # 清除UI上的label显示
        self.label_numer_result.setText("0")
        self.label_time_result.setText('0 s')
        self.label_class_result.setText('None')
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_class_result.setFont(font)
        self.label_score_result.setText("0")  # 显示置信度值
        # 清除位置坐标
        self.label_xmin_result.setText("0")
        self.label_ymin_result.setText("0")
        self.label_xmax_result.setText("0")
        self.label_ymax_result.setText("0")
        self.textEdit_pic.setText('文件未被选中')
        self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                        "border-color: rgb(0, 0, 0);\n"
                                        "color: rgb(0,0,0);\n"
                                        "font: regular 12pt \"华为仿宋\";")
        self.textEdit_camera.setText('实时摄像已关闭')
        self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 0, 0);\n"
                                           "color: rgb(0,0,0);\n"
                                           "font: regular 12pt \"华为仿宋\";")
        self.textEdit_video.setText('实时视频已关闭')
        self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                          "border-color: rgb(0, 0, 0);\n"
                                          "color: rgb(0,0,0);\n"
                                          "font: regular 12pt \"华为仿宋\";")
        self.label_display.clear()
        self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/yyyyy.png);")
        self.flag_timer = ""

        # 调用文件选择对话框
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                "请选取图片文件", getcwd(),  # 起始路径
                                                                "Model File (*.weights)")  # 文件类型
        # 显示提示信息
        if fileName_choose != '':
            self.model_path = fileName_choose
            self.textEdit_model.setText(fileName_choose + ' 已选中')
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/123456.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.toolButton_model.setIcon(icon1)  # 点击后的 图表变换
        else:
            self.model_path = "../yolo-obj/yolov4-tiny.weights"  # 模型默认路径
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/123456.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.toolButton_model.setIcon(icon1)  # 点击后的 图表变换
            self.textEdit_model.setText('使用默认模型')
        self.textEdit_model.setStyleSheet("background-color: transparent;\n"#transparent全透明
                                          "border-color: rgb(0, 0, 0);\n"
                                          "color: rgb(255,0,0);\n"
                                          "font: regular 12pt \"华为仿宋\";")

    def choose_file(self):
        # 选择图片或视频文件后执行此槽函数
        self.pts = [deque(maxlen=30) for _ in range(9999)]
        self.timer_camera.stop()
        self.timer_video.stop()
        if self.cap:
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧

        # 重置下拉选框
        self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
        self.comboBox_select.clear()
        self.comboBox_select.addItem('所有目标')
        self.comboBox_select.currentIndexChanged.connect(self.select_obj)
        # 清除UI上的label显示
        self.label_numer_result.setText("0")
        self.label_time_result.setText('0 s')
        self.label_class_result.setText('None')
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_class_result.setFont(font)
        self.label_score_result.setText("0")  # 显示置信度值
        # 清除位置坐标
        self.label_xmin_result.setText("0")
        self.label_ymin_result.setText("0")
        self.label_xmax_result.setText("0")
        self.label_ymax_result.setText("0")
        self.textEdit_camera.setText('实时摄像已关闭')
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/shexiangji2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_camera.setIcon(icon2)  # 点击后的 图表变换
        self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 0, 0);\n"
                                           "color: rgb(0,0,0);\n"
                                           "font: regular 12pt \"华为仿宋\";")
        self.textEdit_video.setText('实时视频已关闭')
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/shipin1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_video.setIcon(icon1)  # 点击后的 图表变换
        self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                          "border-color: rgb(0, 0, 0);\n"
                                          "color: rgb(0,0,0);\n"
                                          "font: regular 12pt \"华为仿宋\";")
        self.label_display.clear()
        self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/yyyyy.png);")
        self.flag_timer = ""
        # 使用文件选择对话框选择图片
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "选取图片文件",
            self.path,  # 起始路径
            "图片(*.jpg;*.jpeg;*.png)")  # 文件类型
        self.path = fileName_choose  # 保存路径
        if fileName_choose != '':
            self.flag_timer = "image"
            self.textEdit_pic.setText(fileName_choose + '文件已选中')
            self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                            "border-color: rgb(0, 0, 0);\n"
                                            "color: rgb(255,0,0);\n"
                                            "font: regular 12pt \"华为仿宋\";")
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/13.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.toolButton_file.setIcon(icon1)  # 点击后的 图表变换
            self.label_display.setText('正在启动检测系统...\n\nleading')
            QtWidgets.QApplication.processEvents()
            # 生成模型对象

            image = self.cv_imread(fileName_choose)  # 读取选择的图片
            self.current_image = image.copy()

            # 初始化检测模型对象
            detector_model = Detector(model_path=self.model_path)
            KalmanBoxTracker.count = 0
            time_start = time.time()
            self.dets, self.boxes, self.indexIDs, self.cls_IDs = detector_model.run(image)  # 执行检测跟踪
            time_end = time.time()

            # 在UI界面上显示检测结果
            ind_select = 0
            ind = -1
            if len(self.dets) > 0:#det 是在Detector_Model定义的 里面有[x, y, x + w, y + h, confidences[i], classIDs[i]]分别为[0,1,2,3,4,5]
                self.label_class_result.setText(self.LABELS[int(self.dets[ind_select][5])])  # 显示类别 检测是  编号为1的  类别显示的字 例如cup-1 person-1 bick-1
                if len(self.LABELS[int(self.dets[ind_select][5])]) > 7:
                    font = QtGui.QFont()
                    font.setPointSize(12)
                else:
                    font = QtGui.QFont()
                    font.setPointSize(16)
                self.label_class_result.setFont(font)
                self.label_score_result.setText(str(round(self.dets[ind_select][4], 3)))  # 显示置信度值
                # 显示位置坐标
                self.label_xmin_result.setText(str(int(self.dets[ind_select][0])))
                self.label_ymin_result.setText(str(int(self.dets[ind_select][1])))
                self.label_xmax_result.setText(str(int(self.dets[ind_select][2])))
                self.label_ymax_result.setText(str(int(self.dets[ind_select][3])))
            else:
                self.label_class_result.setText("None")
                font = QtGui.QFont()
                font.setPointSize(18)
                self.label_class_result.setFont(font)
                self.label_score_result.setText("0")  # 显示置信度值
                # 显示位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

            if len(self.boxes) > 0:
                for i, box in enumerate(self.boxes):  # 遍历所有标记框
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))

                    center = (int(((box[0]) + (box[2])) / 2), int(((box[1]) + (box[3])) / 2))
                    self.pts[self.indexIDs[i]].append(center)

                    if ind != -1:
                        if ind != i:
                            continue

                    # 在图像上标记目标框
                    color = [int(c) for c in self.COLORS[self.indexIDs[i] % len(self.COLORS)]]
                    cv2.rectangle(image, (x, y), (w, h), color, 4) # 绘制矩形

                    thickness = 5   #是矩形边框线的粗细像素。厚度-1像素将以指定的颜色填充矩形形状。

                    if self.radioButton_countting.isChecked():
                        # 显示某个对象标记框的中心
                        cv2.circle(image, center, 1, color, thickness) #画线 cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
                        # 显示运动轨迹  播放视频的时候 选择counting附加中心点 和轨迹
                        for j in range(1, len(self.pts[self.indexIDs[i]])):
                            if self.pts[self.indexIDs[i]][j - 1] is None or self.pts[self.indexIDs[i]][j] is None:
                                continue
                            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                            cv2.line(image, (self.pts[self.indexIDs[i]][j - 1]), (self.pts[self.indexIDs[i]][j]), color,
                                     thickness)#画直线

                    # 标记跟踪到的目标和数目
                    if self.radioButton_detection.isChecked():
                        text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])],
                                              round(self.dets[len(self.boxes) - i - 1][4], 2))#self.dets[len(self.boxes) - i - 1][4]置信度
                    else:
                        text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])], self.indexIDs[i])#self.indexIDs[i] 编号排列
                    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)#将文字添加到摄像中

                # 更新下拉选框
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                for i in range(len(self.boxes)):
                    text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])], self.indexIDs[i])
                    self.comboBox_select.addItem(text) #把对象加入其中
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                # 更新检测目标数目
                self.label_numer_result.setText(str(max(self.indexIDs)))
                self.label_time_result.setText(str(round((time_end - time_start), 2)) + ' s') #更新检测时间
                QtWidgets.QApplication.processEvents()
            else:
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                # 更新检测目标数目
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0 s')
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                QtWidgets.QApplication.processEvents()

            self.detected_image = image.copy()
            # 在Qt界面中显示检测完成画面


            image = cv2.resize(image, (1152, 648))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_display.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            # 选择取消，恢复界面状态
            self.flag_timer = ""
            self.textEdit_pic.setText('文件未选中')  #文件的操作
            self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                            "border-color: rgb(0, 0, 0);\n"
                                            "color: rgb(0,0,0);\n"
                                            "font: regular 12pt \"华为仿宋\";")
            self.label_display.clear()  # 清除画面
            self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/yyyyy.png);") #恢复原始图片
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/12.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.toolButton_file.setIcon(icon1)  # 点击后的 图表变换
            self.label_class_result.setText('None')
            self.label_time_result.setText('0 s')

    def button_open_camera_click(self):
        if self.timer_video.isActive():
            self.timer_video.stop() #停止视频计时器
        if not self.timer_camera.isActive():  # 检查定时状态
            flag = self.cap.open(self.CAM_NUM)  # 检查相机状态  CAM_NUM 摄像头标号
            if not flag:  # 相机打开失败提示
                msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                                    u"请检测相机与电脑是否连接正确！ ",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            else:
                # 准备运行识别程序
                self.flag_timer = "camera"
                self.textEdit_pic.setText('文件未选中') #文件的操作
                self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                                "border-color: rgb(0, 0, 0);\n"
                                                "color: rgb(0,0,0);\n"
                                                "font: regular 12pt \"华为仿宋\";")
                self.textEdit_video.setText('实时视频已关闭')#视频的操作
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 0, 0);\n"
                                                  "color: rgb(0,0,0);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                QtWidgets.QApplication.processEvents()
                self.textEdit_camera.setText('实时摄像已启动') #摄像头的操作
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 0, 0);\n"
                                                   "color: rgb(255,0,0);\n"
                                                   "font: regular 12pt \"华为仿宋\";")
                icon1 = QtGui.QIcon()
                icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/10.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.toolButton_camera.setIcon(icon1)  # 点击后的 图表变换
                self.label_display.setText('正在启动识别系统...\n\nleading') #正在识别
                # 新建对象
                self.pts = [deque(maxlen=30) for _ in range(9999)]  #deque队列的形式
                self.detector_model = Detector(model_path=self.model_path) #通过模型 得出边框信息
                KalmanBoxTracker.count = 0
                # 重置下拉选框
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0 s')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(18)
                self.label_class_result.setFont(font)
                self.label_score_result.setText("0")  # 显示置信度值
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")
                QtWidgets.QApplication.processEvents()
                # 打开定时器
                self.timer_camera.start(30)
        else:
            # 定时器未开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_camera.stop()
            if self.cap:
                self.cap.release()
            self.label_display.clear()
            self.textEdit_pic.setText('文件未选中')
            self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                            "border-color: rgb(0, 0, 0);\n"
                                            "color: rgb(0,0,0);\n"
                                            "font: regular 12pt \"华为仿宋\";")
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 0, 0);\n"
                                               "color: rgb(0,0,0);\n"
                                               "font: regular 12pt \"华为仿宋\";")
            self.textEdit_video.setText('实时视频已关闭')
            self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                              "border-color: rgb(0, 0, 0);\n"
                                              "color: rgb(0,0,0);\n"
                                              "font: regular 12pt \"华为仿宋\";")
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/shexiangji2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.toolButton_camera.setIcon(icon1)  # 点击后的 图表变换
            self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/yyyyy.png);")

            # 重置下拉选框
            self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
            self.comboBox_select.clear()
            self.comboBox_select.addItem('所有目标')
            self.comboBox_select.currentIndexChanged.connect(self.select_obj)
            # 清除UI上的label显示
            self.label_numer_result.setText("0")
            self.label_time_result.setText('0 s')
            self.label_class_result.setText('None')
            font = QtGui.QFont()
            font.setPointSize(18)
            self.label_class_result.setFont(font)
            self.label_score_result.setText("0")  # 显示置信度值
            # 清除位置坐标
            self.label_xmin_result.setText("0")
            self.label_ymin_result.setText("0")
            self.label_xmax_result.setText("0")
            self.label_ymax_result.setText("0")
            QtWidgets.QApplication.processEvents() #刷新页面

    def show_camera(self):
        # 定时器槽函数，每隔一段时间执行
        flag, image = self.cap.read()  # 获取画面
        if flag:
            image = cv2.flip(image, 1)  # 左右翻转
            self.current_image = image.copy()

            time_start = time.time()  # 计时
            # 使用模型预测
            self.dets, self.boxes, self.indexIDs, self.cls_IDs = self.detector_model.run(image)
            time_end = time.time()
            # 在UI界面上显示检测结果
            # ind = self.comboBox_select.currentIndex() - 1
            # if ind == -1:
            #     ind_select = 0
            # else:
            #     ind_select = len(self.boxes) - ind - 1

            text_select = self.comboBox_select.currentText()
            try:
                if len(self.boxes) > 0:  # 追踪到目标
                    for i, box in enumerate(self.boxes):  # 遍历所有标记框
                        center = (int(((box[0]) + (box[2])) / 2), int(((box[1]) + (box[3])) / 2))
                        self.pts[self.indexIDs[i]].append(center)
                        text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])], self.indexIDs[i])
                        if text_select != "所有目标":
                            if text_select != text:
                                continue

                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))

                        # 在图像上标记目标框
                        color = [int(c) for c in self.COLORS[self.indexIDs[i] % len(self.COLORS)]]
                        cv2.rectangle(image, (x, y), (w, h), color, 4)

                        thickness = 5

                        if self.radioButton_countting.isChecked():
                            # 显示某个对象标记框的中心
                            cv2.circle(image, center, 1, color, thickness)

                            # 显示运动轨迹
                            for j in range(1, len(self.pts[self.indexIDs[i]])):
                                if self.pts[self.indexIDs[i]][j - 1] is None or self.pts[self.indexIDs[i]][j] is None:
                                    continue
                                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                                cv2.line(image, (self.pts[self.indexIDs[i]][j - 1]), (self.pts[self.indexIDs[i]][j]),
                                         color,
                                         thickness)

                        # 标记跟踪到的目标和数目
                        if self.radioButton_detection.isChecked():
                            text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])],
                                                  round(self.dets[len(self.boxes) - i - 1][4], 2))
                        else:
                            text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])], self.indexIDs[i])
                        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                        ind_select = len(self.boxes) - i - 1

                        # 更新界面
                        self.label_class_result.setText(self.LABELS[int(self.cls_IDs[i])])  # 显示类别
                        if len(self.LABELS[int(self.cls_IDs[i])]) > 7:
                            font = QtGui.QFont()
                            font.setPointSize(12)
                        else:
                            font = QtGui.QFont()
                            font.setPointSize(18)
                        self.label_class_result.setFont(font)
                        # 更新界面
                        self.label_score_result.setText(str(round(self.dets[ind_select][4], 3)))  # 显示置信度值
                        # 显示位置坐标
                        self.label_xmin_result.setText(str(int(self.boxes[i][0])))
                        self.label_ymin_result.setText(str(int(self.boxes[i][1])))
                        self.label_xmax_result.setText(str(int(self.boxes[i][2])))
                        self.label_ymax_result.setText(str(int(self.boxes[i][3])))

                    # 更新下拉选框
                    self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                    self.comboBox_select.clear()
                    self.comboBox_select.addItem('所有目标')
                    for j in range(len(self.boxes)):
                        text = "{}-{}".format(self.LABELS[int(self.cls_IDs[j])], self.indexIDs[j])
                        self.comboBox_select.addItem(text)
                    self.comboBox_select.setCurrentText(text_select)  # 设置当前选项
                    self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                    # 更新检测目标数目
                    self.label_numer_result.setText(str(len(self.indexIDs)))
                    self.label_time_result.setText(str(round((time_end - time_start), 2)) + ' s')
                    QtWidgets.QApplication.processEvents()
                else:
                    self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                    self.comboBox_select.clear()
                    self.comboBox_select.addItem('所有目标')
                    # 更新检测目标数目
                    self.label_numer_result.setText("0")
                    self.label_time_result.setText('0 s')
                    self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                    QtWidgets.QApplication.processEvents()

                # 是否检测到目标
                if len(self.dets) == 0:
                    self.label_class_result.setText("None")
                    font = QtGui.QFont()
                    font.setPointSize(18)
                    self.label_class_result.setFont(font)
                    self.label_score_result.setText("0")  # 显示置信度值
                    # 显示位置坐标
                    self.label_xmin_result.setText("0")
                    self.label_ymin_result.setText("0")
                    self.label_xmax_result.setText("0")
                    self.label_ymax_result.setText("0")
            except:
                print("[INFO] 目标消失，请重新选择目标选项！")

            # 在Qt界面中显示检测完成画面
            image = cv2.resize(image, (1152, 648))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_display.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.timer_camera.stop()

    def button_open_video_click(self): #打开视频
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        if not self.timer_video.isActive():  # 检查定时状态
            # 弹出文件选择框选择视频文件
            fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget, "选取视频文件",
                                                                    self.video_path,  # 起始路径
                                                                    "视频(*.mp4;*.avi)")  # 文件类型
            self.video_path = fileName_choose

            if fileName_choose != '':
                self.flag_timer = "video"
                self.textEdit_video.setText(fileName_choose + '文件已选中')
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 0, 0);\n"
                                                  "color: rgb(255,0,0);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                icon1 = QtGui.QIcon()
                icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/11.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.toolButton_video.setIcon(icon1)  # 点击后的 图表变换
                self.label_display.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()

                try:  # 初始化视频流
                    self.cap_video = cv2.VideoCapture(fileName_choose)
                except:
                    print("[INFO] could not determine # of frames in video")
                # 准备运行识别程序
                self.textEdit_pic.setText('文件未选中')
                self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                                "border-color: rgb(0, 0, 0);\n"
                                                "color: rgb(0, 0, 0);\n"
                                                "font: regular 12pt \"华为仿宋\";")
                QtWidgets.QApplication.processEvents()
                self.textEdit_camera.setText('实时摄像未启动')
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 0, 0);\n"
                                                   "color: rgb(0, 0, 0);\n"
                                                   "font: regular 12pt \"华为仿宋\";")
                self.textEdit_video.setText(fileName_choose + '文件已选中')
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 0, 0);\n"
                                                  "color: rgb(255, 0, 0);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                self.label_display.setText('正在启动识别系统...\n\nleading')
                # 新建对象
                self.pts = [deque(maxlen=30) for _ in range(9999)]
                self.detector_model = Detector(model_path=self.model_path)
                KalmanBoxTracker.count = 0
                # 重置下拉选框
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有目标')
                self.comboBox_select.setCurrentText("所有目标")  # 设置当前选项
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0 s')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(20)
                self.label_class_result.setFont(font)
                self.label_score_result.setText("0")  # 显示置信度值
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")
                QtWidgets.QApplication.processEvents()
                # 打开定时器
                self.timer_video.start(30)

            else:
                # 选择取消，恢复界面状态
                self.flag_timer = ""
                self.textEdit_pic.setText('文件未选中')
                icon1 = QtGui.QIcon()
                icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/12.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.toolButton_file.setIcon(icon1)  # 点击后的 图表变换
                self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                                "border-color: rgb(0, 0, 0);\n"
                                                "color: rgb(0,0,0);\n"
                                                "font: regular 12pt \"华为仿宋\";")
                self.textEdit_camera.setText('实时摄像已关闭')
                icon1 = QtGui.QIcon()
                icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/shexiangji2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.toolButton_camera.setIcon(icon1)  # 点击后的 图表变换
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 0, 0);\n"
                                                   "color: rgb(0,0,0);\n"
                                                   "font: regular 12pt \"华为仿宋\";")
                self.textEdit_video.setText('实时视频未选中')
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 0, 0);\n"
                                                  "color: rgb(0,0,0);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                self.label_display.clear()  # 清除画面
                self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/yyyyy.png);")
                icon1 = QtGui.QIcon()
                icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/shipin1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                self.toolButton_video.setIcon(icon1)  # 点击后的 图表变换
                self.label_class_result.setText('None')
                self.label_time_result.setText('0 s')

        else:
            # 定时器未开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_video.stop()
            self.cap_video.release()
            self.label_display.clear()
            self.textEdit_pic.setText('文件未选中')
            self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                            "border-color: rgb(0, 0, 0);\n"
                                            "color: rgb(0,0,0);\n"
                                            "font: regular 12pt \"华为仿宋\";")
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 0, 0);\n"
                                               "color: rgb(0,0,0);\n"
                                               "font: regular 12pt \"华为仿宋\";")
            self.textEdit_video.setText('实时视频已关闭')
            self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                              "border-color: rgb(0, 0, 0);\n"
                                              "color: rgb(0,0,0);\n"
                                              "font: regular 12pt \"华为仿宋\";")

            self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/yyyyy.png);")
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/shipin1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.toolButton_video.setIcon(icon1)  # 点击后的 图表变换

            # 重置下拉选框
            self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
            self.comboBox_select.clear()
            self.comboBox_select.addItem('所有目标')
            self.comboBox_select.currentIndexChanged.connect(self.select_obj)
            # 清除UI上的label显示
            self.label_numer_result.setText("0")
            self.label_time_result.setText('0 s')
            self.label_class_result.setText('None')
            font = QtGui.QFont()
            font.setPointSize(18)
            self.label_class_result.setFont(font)
            self.label_score_result.setText("0")  # 显示置信度值
            # 清除位置坐标
            self.label_xmin_result.setText("0")
            self.label_ymin_result.setText("0")
            self.label_xmax_result.setText("0")
            self.label_ymax_result.setText("0")
            QtWidgets.QApplication.processEvents()

    def show_video(self):
        # 定时器槽函数，每隔一段时间执行
        flag, image = self.cap_video.read()  # 获取画面
        if flag:
            self.current_image = image.copy()

            time_start = time.time()  # 计时
            # 使用模型预测
            self.dets, self.boxes, self.indexIDs, self.cls_IDs = self.detector_model.run(image)
            time_end = time.time()
            # 在UI界面上显示检测结果
            # ind = self.comboBox_select.currentIndex() - 1
            # if ind == -1:
            #     ind_select = 0
            # else:
            #     ind_select = len(self.boxes) - ind - 1

            text_select = self.comboBox_select.currentText()
            try:
                if len(self.boxes) > 0:  # 追踪到目标
                    for i, box in enumerate(self.boxes):  # 遍历所有标记框

                        center = (int(((box[0]) + (box[2])) / 2), int(((box[1]) + (box[3])) / 2))
                        self.pts[self.indexIDs[i]].append(center)

                        text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])], self.indexIDs[i])
                        if text_select != "所有目标":
                            if text_select != text:
                                continue
                        # if ind != -1:
                        #     if ind != i:
                        #         continue

                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))

                        # 在图像上标记目标框
                        color = [int(c) for c in self.COLORS[self.indexIDs[i] % len(self.COLORS)]]
                        cv2.rectangle(image, (x, y), (w, h), color, 4)

                        thickness = 5

                        if self.radioButton_countting.isChecked():
                            # 显示某个对象标记框的中心
                            cv2.circle(image, center, 1, color, thickness)

                            # 显示运动轨迹
                            for j in range(1, len(self.pts[self.indexIDs[i]])):
                                if self.pts[self.indexIDs[i]][j - 1] is None or self.pts[self.indexIDs[i]][j] is None:
                                    continue
                                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                                cv2.line(image, (self.pts[self.indexIDs[i]][j - 1]), (self.pts[self.indexIDs[i]][j]),
                                         color,
                                         thickness)

                        # 标记跟踪到的目标和数目
                        if self.radioButton_detection.isChecked():
                            text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])],
                                                  round(self.dets[len(self.boxes) - i - 1][4], 2))
                        else:
                            text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])], self.indexIDs[i])
                        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                        ind_select = len(self.boxes) - i - 1
                        # 更新界面
                        self.label_class_result.setText(self.LABELS[int(self.cls_IDs[i])])  # 显示类别
                        if len(self.LABELS[int(self.cls_IDs[i])]) > 7:
                            font = QtGui.QFont()
                            font.setPointSize(12)
                        else:
                            font = QtGui.QFont()
                            font.setPointSize(18)
                        self.label_class_result.setFont(font)
                        # 更新界面
                        self.label_score_result.setText(str(round(self.dets[ind_select][4], 3)))  # 显示置信度值
                        # 显示位置坐标
                        self.label_xmin_result.setText(str(int(self.boxes[i][0])))
                        self.label_ymin_result.setText(str(int(self.boxes[i][1])))
                        self.label_xmax_result.setText(str(int(self.boxes[i][2])))
                        self.label_ymax_result.setText(str(int(self.boxes[i][3])))

                    # 更新下拉选框
                    self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                    self.comboBox_select.clear()
                    self.comboBox_select.addItem('所有目标')
                    for j in range(len(self.boxes)):
                        text = "{}-{}".format(self.LABELS[int(self.cls_IDs[j])], self.indexIDs[j])
                        self.comboBox_select.addItem(text)
                    # if self.comboBox_select.currentText() != "":
                    #     self.comboBox_select.setCurrentIndex(ind + 1)  # 设置当前选项
                    self.comboBox_select.setCurrentText(text_select)  # 设置当前选项
                    self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                    # 更新检测目标数目
                    self.label_numer_result.setText(str(len(self.indexIDs)))
                    self.label_time_result.setText(str(round((time_end - time_start), 2)) + ' s')
                    QtWidgets.QApplication.processEvents()
                else:
                    self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                    self.comboBox_select.clear()
                    self.comboBox_select.addItem('所有目标')
                    # 更新检测目标数目
                    self.label_numer_result.setText("0")
                    self.label_time_result.setText('0 s')
                    self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                    QtWidgets.QApplication.processEvents()

                # 是否检测到目标
                if len(self.dets) == 0:
                    # 更新界面
                    # self.label_class_result.setText(self.LABELS[int(self.dets[ind_select][5])])  # 显示类别
                    # if len(self.LABELS[int(self.dets[ind_select][5])]) > 7:
                    #     font = QtGui.QFont()
                    #     font.setPointSize(12)
                    # else:
                    #     font = QtGui.QFont()
                    #     font.setPointSize(18)
                    # self.label_class_result.setFont(font)
                    # # 更新界面
                    # self.label_score_result.setText(str(round(self.dets[ind_select][4], 3)))  # 显示置信度值
                    # # 显示位置坐标
                    # self.label_xmin_result.setText(str(int(self.dets[ind_select][0])))
                    # self.label_ymin_result.setText(str(int(self.dets[ind_select][1])))
                    # self.label_xmax_result.setText(str(int(self.dets[ind_select][2])))
                    # self.label_ymax_result.setText(str(int(self.dets[ind_select][3])))
                    # else:  # 无目标则清除
                    self.label_class_result.setText("None")
                    font = QtGui.QFont()
                    font.setPointSize(18)
                    self.label_class_result.setFont(font)
                    self.label_score_result.setText("0")  # 显示置信度值
                    # 显示位置坐标
                    self.label_xmin_result.setText("0")
                    self.label_ymin_result.setText("0")
                    self.label_xmax_result.setText("0")
                    self.label_ymax_result.setText("0")
            except:
                print("[INFO] 目标消失，请重新选择目标选项！")

            # 在Qt界面中显示检测完成画面
            image = cv2.resize(image, (1152, 648))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_display.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()

    def select_obj(self):
        QtWidgets.QApplication.processEvents()
        if self.flag_timer == "video":
            # 打开定时器
            self.timer_video.start(30)
        elif self.flag_timer == "camera":
            self.timer_camera.start(30)

        ind = self.comboBox_select.currentIndex() - 1
        if ind == -1:
            ind_select = 0
        else:
            ind_select = len(self.boxes) - ind - 1
        if len(self.dets) > 0:
            if len(self.LABELS[int(self.dets[ind_select][5])]) > 7:  #类别的名字比7长
                font = QtGui.QFont()
                font.setPointSize(12)
            else:
                font = QtGui.QFont()
                font.setPointSize(18)
            self.label_class_result.setFont(font)
            self.label_class_result.setText(self.LABELS[int(self.dets[ind_select][5])])  # 显示类别
            self.label_score_result.setText(str(round(self.dets[ind_select][4], 2)))  # 显示置信度值
            # 显示位置坐标
            self.label_xmin_result.setText(str(int(self.dets[ind_select][0])))
            self.label_ymin_result.setText(str(int(self.dets[ind_select][1])))
            self.label_xmax_result.setText(str(int(self.dets[ind_select][2])))
            self.label_ymax_result.setText(str(int(self.dets[ind_select][3])))

        image = self.current_image.copy()
        if len(self.boxes) > 0:
            for i, box in enumerate(self.boxes):  # 遍历所有标记框
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                if ind != -1:
                    if ind != i:
                        continue
                # 在图像上标记目标框
                color = [int(c) for c in self.COLORS[self.indexIDs[i] % len(self.COLORS)]]
                cv2.rectangle(image, (x, y), (w, h), color, 4)

                center = (int(((box[0]) + (box[2])) / 2), int(((box[1]) + (box[3])) / 2))
                self.pts[self.indexIDs[i]].append(center)
                thickness = 5

                if self.radioButton_countting.isChecked():
                    # 显示某个对象标记框的中心
                    cv2.circle(image, center, 1, color, thickness)

                    # 显示运动轨迹
                    for j in range(1, len(self.pts[self.indexIDs[i]])):
                        if self.pts[self.indexIDs[i]][j - 1] is None or self.pts[self.indexIDs[i]][j] is None:
                            continue
                        thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                        cv2.line(image, (self.pts[self.indexIDs[i]][j - 1]), (self.pts[self.indexIDs[i]][j]), color,
                                 thickness)
                # 标记跟踪到的目标和数目
                if self.radioButton_detection.isChecked():
                    text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])],
                                          round(self.dets[len(self.boxes) - i - 1][4], 2))
                else:
                    text = "{}-{}".format(self.LABELS[int(self.cls_IDs[i])], self.indexIDs[i])
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # 在Qt界面中显示检测完成画面
        image = cv2.resize(image, (1152, 648))  # 设定图像尺寸为显示界面大小
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_display.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def cv_imread(self, filePath):
        # 读取图片
        # cv_img = cv2.imread(filePath)
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        ## cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        if cv_img.shape[2] > 3:
            cv_img = cv_img[:, :, :3]
        return cv_img

    def save_file(self):
        if self.detected_image is not None:
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/15.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.toolButton_saveing.setIcon(icon1)  # 点击后的 图表变换
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            cv2.imwrite('./pic_' + str(now_time) + '.png', self.detected_image)
            QMessageBox.about(self.centralwidget, "保存文件", "\nSuccessed!\n文件已保存！")
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/baocun2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.toolButton_saveing.setIcon(icon1)  # 点击后的 图表变换

        else:
            QMessageBox.about(self.centralwidget, "保存文件", "saving...\nFailed!\n请先进行检测操作！")
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/newPrefix/images_test/baocun2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.toolButton_saveing.setIcon(icon1)  # 点击后的 图表变换

    def setting(self):
        QMessageBox.about(self.centralwidget, "图片默认保存", "保存位置为当前文件夹路径\nHmx")

    def disp_version(self):
        QMessageBox.about(self.centralwidget, "版本信息", "行人车辆多物体检测与跟踪软件\nv1.0\nHmx")

    def disp_website(self):
        QMessageBox.about(self.centralwidget, "应用主旨", "为国家安防领域的发事业献身\nHmx\n")

