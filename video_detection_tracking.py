# -*- coding: utf-8 -*-
# 本程序用于视频中车辆行人等多目标检测跟踪
# @Time    : 2021/6
# @Author  : yangyang
# @Email   : yy2289150297@163.com
# @Software: PyCharm

import warnings
from collections import deque
import numpy as np
import imutils
import time
import cv2
import os
from tqdm import tqdm
from sort import Sort

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    # 参数设置
    video_path = "./video/pedestrian.mp4"  # 要检测的视频路径
    filter_confidence = 0.5  # 用于筛除置信度过低的识别结果
    threshold_prob = 0.3  # 用于NMS去除重复的锚框
    model_path = "./yolo-obj"  # 模型文件的目录

    # 载入数据集标签
    labelsPath = os.path.sep.join([model_path, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # 载入模型参数文件及配置文件
    weightsPath = os.path.sep.join([model_path, "yolov4.weights"])
    configPath = os.path.sep.join([model_path, "yolov4.cfg"])

    # 初始化用于标记框的颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    # 用于展示目标移动路径
    pts = [deque(maxlen=30) for _ in range(9999)]

    # 从配置和参数文件中载入模型
    print("[INFO] 正在载入模型...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 初始化视频流
    vs = cv2.VideoCapture(video_path)
    (W, H) = (None, None)
    frameIndex = 0  # 视频帧数

    # 试运行，获取总的画面帧数
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))

    # 若读取失败，报错退出
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = vs.read()
    vw = frame.shape[1]
    vh = frame.shape[0]
    print("[INFO] 视频尺寸：{} * {}".format(vw, vh))
    output_video = cv2.VideoWriter(video_path.replace(".mp4", "-det.avi"), fourcc, 20.0, (vw, vh))  # 处理后的视频对象

    tracker = Sort()  # 实例化追踪器对象

    # 遍历视频帧进行检测
    for fr in tqdm(range(total)):
        # 从视频文件中逐帧读取画面
        (grabbed, frame) = vs.read()

        # 若grabbed为空，表示视频到达最后一帧，退出
        if not grabbed:
            break

        # 获取画面长宽
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # 将一帧画面读入网络
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        boxes = []  # 用于检测框坐标
        confidences = []  # 用于存放置信度值
        classIDs = []  # 用于识别的类别序号

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
                if confidence > filter_confidence:
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
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, filter_confidence, threshold_prob)

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
        tracks = tracker.update(dets)

        boxes = []  # 存放追踪到的标记框
        indexIDs = []
        cls_IDs = []
        c = []

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            cls_IDs.append(int(track[5]))

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:  # 遍历所有标记框
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # 在图像上标记目标框
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x, y), (w, h), color, 4)

                center = (int(((box[0]) + (box[2])) / 2), int(((box[1]) + (box[3])) / 2))
                pts[indexIDs[i]].append(center)
                thickness = 5
                # 显示某个对象标记框的中心
                cv2.circle(frame, center, 1, color, thickness)

                # 显示目标运动轨迹
                for j in range(1, len(pts[indexIDs[i]])):
                    if pts[indexIDs[i]][j - 1] is None or pts[indexIDs[i]][j] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(frame, (pts[indexIDs[i]][j - 1]), (pts[indexIDs[i]][j]), color, thickness)

                # 标记跟踪到的目标和数目
                text = "{}-{}".format(LABELS[int(cls_IDs[i])], indexIDs[i])
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                i += 1

        # 实时显示检测画面
        cv2.imshow('Stream', frame)
        output_video.write(frame)  # 保存标记后的视频
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # print("FPS:{}".format(int(0.6/(end-start))))
        frameIndex += 1

        if frameIndex >= total:  # 可设置检测的最大帧数提前退出
            print("[INFO] 运行结束...")
            output_video.release()
            vs.release()
            exit()
