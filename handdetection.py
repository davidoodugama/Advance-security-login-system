from random import random
import cv2
import time
from datetime import datetime
import os
import Switcher
import random
import HandTrackingModule as htm
wCam , hCam = 648, 468
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
checkFinger = Switcher.PythonSwitch()
detector = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]
random_number = random.randint(1, 5)
time_count = 0
timer_counter = 0
start_time = 0
duration = 2

def setTimer(timer_counter, start_time):
    timer_counter += 1
    start_time = datetime.now()
    timer = True
    return start_time, timer, timer_counter

while True:
    ret, img = cap.read()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    img, landMarks = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    cv2.putText(img, "Show finger number:" + str(random_number), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    if timer_counter == 0:
            start_time, timer, timer_counter = setTimer(timer_counter, start_time)
    if timer:
        end_time = time.time()
        diff = (datetime.now() - start_time).seconds
        cv2.putText(img, "Count Down" + str(diff), (0, 300), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
        if diff >= duration:
            timer = False
    if len(lmList) != 0:
        fingers = []
        
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # if timer_counter == 0:
        #     start_time, timer, timer_counter = setTimer(timer_counter, start_time)
        if landMarks != None:
            if timer:
                res = checkFinger.CheckShwonFinger(random_number, fingers)
                if res:
                    cv2.putText(img, "User authenticated as dynamic", (0, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                else:
                    cv2.putText(img, "User not authenticated as dynamic", (0, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                # end_time = time.time()
                # diff = (datetime.now() - start_time).seconds
                # cv2.putText(img, "Count Down" + str(diff), (0, 300), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    
    cv2.imshow("image", img)
    cv2.waitKey(1)
