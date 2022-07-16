import cv2 as cv
import face_recognition
import os
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import Switcher
from random import random
import time
from datetime import datetime
import os
import random
import HandTrackingModule as htm
from cvzone import stackImages, putTextRect
path = "images"
images = []
classNames = []
kernel = np.array([[1, 0, 0],
                   [0, 4, -1],
                   [-1, -1, 0]])
myList = os.listdir(path)
for cls in myList:
    current_img = cv.imread(f'{path}/{cls}')
    current_img = cv.resize(current_img, (800,800), interpolation = cv.INTER_AREA)
    frame = cv.filter2D(src = current_img, ddepth = -1, kernel = kernel)
    current_img = cv.GaussianBlur(current_img, (3,3), cv.BORDER_DEFAULT)
    images.append(current_img)
    classNames.append(os.path.splitext(cls)[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        faceLocCurrentFrame = face_recognition.face_locations(img)
        for y1,x2,y2,x1 in faceLocCurrentFrame:
            face = img[y1:y1+y2, x1:x1+x2, :]
        encode = face_recognition.face_encodings(face)[0]
        encodeList.append(encode)

    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding complete')

cap = cv.VideoCapture(0)
detector = FaceMeshDetector(maxFaces = 1)
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW = [ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW = [ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
# idList = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
plotY = LivePlot(800,800,[20,50])
ratioList = []
rightRatioList = []
CEF_COUNTER = 0
CLOSED_EYES_FRAME = 4
TOTAL_BLINKS = 0
color = (255, 0, 255)
# Face Detector
fac_detector = FaceDetector()

# Hand Detection variables
pTime = 0
checkFinger = Switcher.PythonSwitch()
hand_detector = htm.handDetector()
tipIds = [4, 8, 12, 16, 20]
random_number = random.randint(1, 5)
time_count = 0
timer_counter = 0
start_time = 0
duration = 60

def setTimer(timer_counter, start_time):
    timer_counter += 1
    start_time = datetime.now()
    timer = True
    return start_time, timer, timer_counter
    

while True:
    ret, frame = cap.read()
    img, faces = detector.findFaceMesh(frame, draw = False)
    # Hand Detection
    hand_img, landMarks = hand_detector.findHands(frame, draw=True)
    lmList = hand_detector.findPosition(hand_img, draw=False)

    if faces:
        detected_face = faces[0]
        pointLeft = detected_face[145]
        pointRight = detected_face[374]
        cv.line(img, pointLeft, pointRight, (0,200,0), 3)
        cv.circle(img, pointLeft, 5, (255,0,255), cv.FILLED)
        cv.circle(img, pointRight, 5, (255,0,255), cv.FILLED)
        smallW, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3 # Males 6.3cm, Females 6.2cm

        # finding the focal length
        # distance = 50
        # f = (smallW * distance)/ W

        # Finding the distance of the actual face
        f = 692
        D = int((W * f) / smallW)
        putTextRect(img, f'Depth: {int(D)}cm', 
                    (detected_face[10][0]-75,detected_face[10][1]-50),
                    scale = 2)
        # FACE DISTANCE
        if D in range(45, 55):
            if faces:
                face = faces[0]
                for id in RIGHT_EYE:
                    cv.circle(img, face[id], 5, color, cv.FILLED)
                for id in LEFT_EYE:
                    cv.circle(img, face[id], 5, color, cv.FILLED)
                
                # LEFT EYE
                leftUp = face[159]
                leftDown = face[23]
                leftLeft = face[130]
                leftRight = face[243]
                # RIGHT
                rightUp = face[385]
                rightDown = face[374]
                rightLeft = face[263]
                rightRight = face[362]

                # FROM TOP OF THE LEFT EYE TO THE BOTTOM THE LENGTH
                lengthVer,_ = detector.findDistance(leftUp, leftDown)
                # FROM LEFT OF THE EYE TO THE RIGHT SIDE OF THE LEFT EYE THE LENGTH
                lengthHor,_ = detector.findDistance(leftLeft, leftRight)
                # FROM TOP OF THE RIGHT EYE TO THE BOTTOM THE LENGTH
                RightlengthVer,_ = detector.findDistance(rightUp, rightDown)
                # FROM RIGHT OF THE EYE TO THE RIGHT SIDE OF THE RIGHT EYE THE LENGTH
                RightlengthHor,_ = detector.findDistance(rightLeft, rightRight)
                # DRAW LINES FOR THE ABOVE
                cv.line(img, leftUp, leftDown, (0,200,0), 3)
                cv.line(img, leftLeft, leftRight, (0,200,0), 3)
                cv.line(img, rightUp, rightDown, (0,200,0), 3)
                cv.line(img, rightLeft, rightRight, (0,200,0), 3)
                # GETTING THE RATIO OF THE BLINK
                Rightratio = int((RightlengthVer/RightlengthHor)* 100)
                ratio = int((lengthVer/lengthHor)* 100)
                ratioList.append(ratio)
                rightRatioList.append(Rightratio)
                # ONLY KEEP 3 VALUES IN THE LIST
                if len(ratioList) > 3 and len(rightRatioList) > 3: 
                    ratioList.pop(0)
                    rightRatioList.pop(0)
                ratioAvg = sum(ratioList)/len(ratioList)
                RightratioAvg = sum(rightRatioList)/len(rightRatioList)
            
                if ratioAvg > 34 and RightratioAvg > 29:
                    CEF_COUNTER += 1
                    color = (255, 0, 255)
                else:
                    if CEF_COUNTER>CLOSED_EYES_FRAME:
                        TOTAL_BLINKS += 1
                        color = (0, 200, 0)
                        CEF_COUNTER = 0
                putTextRect(img, f'Blink Count: {TOTAL_BLINKS}', (detected_face[10][0]-100,detected_face[10][1]-100), colorR = color)
                if TOTAL_BLINKS == 5:
                    TOTAL_BLINKS = 5
                    frame = cv.filter2D(src = frame, ddepth = -1, kernel = kernel)
                    frame = cv.GaussianBlur(frame, (3,3), cv.BORDER_DEFAULT)
                    img = cv.resize(frame, (0,0), None, 0.50, 0.50)
                    imgCurrentFrame = cv.cvtColor(img , cv.COLOR_BGR2RGB)
                    faceLocCurrentFrame = face_recognition.face_locations(imgCurrentFrame)
                    encodeCurrentFrame = face_recognition.face_encodings(imgCurrentFrame, faceLocCurrentFrame)

                    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceLocCurrentFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)   
                        matchIndex = np.argmin(faceDis)

                        if matches[matchIndex]:
                            cv.putText(frame, "Show finger number:" + str(random_number), (0, 30), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
                            name = classNames[matchIndex].upper()
                            print(name)
                            y1,x2,y2,x1 = faceLoc
                            cv.rectangle(frame, (x1*2, y1*2),(x2*2, y2*2), (0,255,0),2)
                            cv.rectangle(frame, (x1*2, y2*2-25),(x2*2, y2*2),(0,255,0), cv.FILLED)
                            cv.putText(frame, name, (x1*2 +6, y2*2-2), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                            if timer_counter == 0:
                                start_time, timer, timer_counter = setTimer(timer_counter, start_time)
                            if timer:
                                end_time = time.time()
                                diff = (datetime.now() - start_time).seconds
                                cv.putText(frame, "Count Down" + str(diff), (0, 70), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
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
                                            cv.putText(frame, "User authenticated as dynamic", (0, 100), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
                                        else:
                                            cv.putText(frame, "User not authenticated as dynamic", (0, 100), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

            else:
                img = cv.resize(frame, (1000,1000), interpolation = cv.INTER_AREA)
    cv.imshow('face', frame)
    cv.waitKey(1)