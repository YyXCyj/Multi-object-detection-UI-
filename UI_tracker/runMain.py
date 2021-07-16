# -*- coding: utf-8 -*-
"""
运行本项目需要安装的库：
    opencv-contrib-python 4.5.1.48
    PyQt5 5.15.2
    scikit-learn 0.22
    numba 0.53.0
    imutils 0.5.4
    filterpy 1.4.5
"""
import os
import warnings
from DetectionTracking import Ui_MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    # 忽略警告
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings(action='ignore')
    app = QApplication(argv)

    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())
