# -*- coding: utf-8 -*-
# 本程序用于视频中车辆行人等多目标检测跟踪
# @Time    : 2021/6
# @Author  : yangyang
# @Email   : yy2289150297@163.com
# @Software: PyCharm

import os
import time

import cv2
import numpy as np

from sort import Sort


class Detector:
    def __init__(self, model_path=None, video_path=None):
        self.filter_confidence = 0.5  # 用于筛除置信度过低的识别结果
        self.threshold_prob = 0.3  # 用于NMS去除重复的锚框

        if model_path is None:
            model_path = "../yolo-obj"  # 模型文件的目录

        if video_path is None:
            video_path = '../video/JapanStreet.mp4'
        # 载入模型参数文件及配置文件
        # weightsPath = os.path.sep.join([model_path, "yolov4-tiny.weights"])
        # configPath = os.path.sep.join([model_path, "yolov4-tiny.cfg"])
        weightsPath = "".join(model_path)
        configPath = model_path.replace("weights", "cfg")

        # 载入数据集标签
        labelsPath = os.path.sep.join(["../yolo-obj", "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")# 数据集排列
        self.net = None
        # 从配置和参数文件中载入模型
        try:
            self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
            ln = self.net.getLayerNames()
            self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            self.tracker = Sort()  # 实例化追踪器对象
        except:
            print("读取模型失败，请检查文件路径并确保无中文文件夹！")

    def run(self, frame):
        frame_in = frame.copy()
        # 将一帧画面读入网络
        blob = cv2.dnn.blobFromImage(frame_in, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()

        boxes = []  # 用于检测框坐标
        confidences = []  # 用于存放置信度值
        classIDs = []  # 用于识别的类别序号

        (H, W) = frame_in.shape[:2]

        # 逐层遍历网络获取输出
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # 过滤低置信度值的检测结果
                if confidence > self.filter_confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # 转换标记框
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # 更新标记框、置信度值、类别列表
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # 使用NMS去除重复的标记框
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.filter_confidence, self.threshold_prob)

        dets = []
        if len(idxs) > 0:
            # 遍历索引得到检测结果
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x + w, y + h, confidences[i], classIDs[i]])

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)

        # 使用sort算法，开始进行追踪
        tracks = self.tracker.update(dets)
        boxes = []  # 存放追踪到的标记框
        indexIDs = []  # 存放追踪到的序号
        cls_IDs = []  # 存放追踪到的类别

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            cls_IDs.append(int(track[5]))

        return dets, boxes, indexIDs, cls_IDs

