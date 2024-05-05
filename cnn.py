import numpy as np
import time
import pygame
import cv2 as cv
import dlib
from scipy.spatial import distance
from eyesCNN import predict
from keras.models import load_model
import os

# 调用人脸检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("68_face_landmarks.dat")

# 设定人眼标定点
LeftEye_Start = 36
LeftEye_End = 41
RightEye_Start = 42
RightEye_End = 47

Radio = 0.2  # 横纵比阈值
Low_radio_constant = 2  # 当Radio小于阈值时，接连多少帧一定发生眨眼动作


def calculate_Ratio(eye):
    """
    计算眼睛横纵比
    """
    d1 = distance.euclidean(eye[1], eye[5])
    d2 = distance.euclidean(eye[2], eye[4])
    d3 = distance.euclidean(eye[0], eye[3])
    d4 = (d1 + d2) / 2
    ratio = d4 / d3
    return ratio


def main():
    """
    主函数
    """
    blink_counter = 0  # 眨眼计数
    frame_counter = 0  # 连续帧计数
    time_start = time.time()
    cap = cv.VideoCapture(0)  # 0摄像头摄像
    cv.waitKey(5)
    model = load_model('eyesCNN.h5')
    while cap.isOpened():
        ret, frame = cap.read()  # 读取每一帧
        cv.waitKey(5)
        # if ret:  # 若读取到图像再进行显示
        #     cv.imshow('winName', frame)
        #     cv.waitKey()
        frame = cv.flip(frame, 1)
        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            rects = detector(gray, 0)  # 人脸检测

            for rect in rects:
                # 提取人脸区域的坐标
                left = rect.left()
                top = rect.top()
                right = rect.right()
                bottom = rect.bottom()
                # 切割出人脸图像
                face_img = frame[top:bottom, left:right]
                im = cv.resize(face_img,(50,50))
                im = np.dot(np.array(im, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
                im = np.expand_dims(im, axis=0)
                im = np.transpose(im, (0, 3, 1, 2))  # 转换维度顺序
                result = predict(model, im)

                shape = predictor(gray, rect)
                points = np.zeros((68, 2), dtype=int)
                for i in range(68):
                    points[i] = (shape.part(i).x, shape.part(i).y)

                # 获取眼睛特征点
                Lefteye = points[LeftEye_Start: LeftEye_End + 1]
                Righteye = points[RightEye_Start: RightEye_End + 1]

                # 计算眼睛横纵比
                Lefteye_Ratio = calculate_Ratio(Lefteye)
                Righteye_Ratio = calculate_Ratio(Righteye)
                mean_Ratio = (Lefteye_Ratio + Righteye_Ratio) / 2  # 计算两眼平均比例

                # 计算凸包
                left_eye_hull = cv.convexHull(Lefteye)
                right_eye_hull = cv.convexHull(Righteye)

                # 绘制轮廓
                cv.drawContours(frame, [left_eye_hull], -1, [0, 255, 0], 1)
                cv.drawContours(frame, [right_eye_hull], -1, [0, 255, 0], 1)

                # 眨眼判断
                if mean_Ratio < Radio:
                    frame_counter += 1
                else:
                    if frame_counter >= Low_radio_constant:
                        blink_counter += 1
                        frame_counter = 0

                # COUNTER = 0
                # TOTAL=0
                # alarm = False
                # cap = cv.VideoCapture(0)
                # start = time.time()
                # if mean_Ratio < Radio:  # 眼睛长宽比：0.2
                #       COUNTER += 1
                #       if COUNTER == 15:
                #        if not alarm:
                #         alarm = True
                #       cv.putText(frame, "wake up!!!", (200, 200),cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                #       if COUNTER > 15:
                #              file = r'警报音乐.mp3'
                #              pygame.init()
                #              print("警报！！！请保持清醒状态驾驶")
                #              track = pygame.mixer.music.load(file)
                #              pygame.mixer.music.play()
                #              time.sleep(14)
                #              pygame.mixer.music.stop()
                #              break
                #              alarm = False
                #       #如果连续 2 次都小于阈值，则表示进行了一次眨眼活动
                #       if COUNTER >= Low_radio_constant:  # 阈值：2
                #           TOTAL += 1
                #           COUNTER = 0
                #
                # 显示结果
                cv.putText(frame, "Eshausted{}".format(blink_counter), (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)
                cv.putText(frame, "Ratio{:.2f}".format(mean_Ratio), (300, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 255], 2)



            cv.imshow("眨眼检测", frame)
            # 计算经过的时间
            elapsed_time = time.time() - time_start
            if elapsed_time >= 15:
                if blink_counter >7:
                    file = r'警报音乐.mp3'
                    pygame.init()
                    print("警报！！！请保持清醒状态驾驶")
                    track = pygame.mixer.music.load(file)
                    pygame.mixer.music.play()
                    time.sleep(4)
                    pygame.mixer.music.stop()
                    break
                # 重置计数器和时间
                blink_counter = 0
                time_start = time.time()

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
