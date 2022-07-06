import cv2 as cv
import face_recognition
import os
import numpy as np
import cvzone

path = "images"
images = []
classNames = []
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
myList = os.listdir(path)
for cls in myList:
    current_img = cv.imread(f'{path}/{cls}')
    current_img = cv.resize(current_img, (800,800), interpolation = cv.INTER_AREA)
    # current_img = cv.filter2D(src = current_img, ddepth = -1, kernel = kernel)
    # current_img = cv.GaussianBlur(current_img, (3,3), cv.BORDER_DEFAULT)
    # current_img = cv.resize(current_img, (740,800), interpolation = cv.INTER_AREA)
    images.append(current_img)
    classNames.append(os.path.splitext(cls)[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        # face_casecade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        # faces = face_casecade.detectMultiScale(img)
        # for x,y,w,h in faces:
        #     face = img[y:y+h+80, x:x+w, :]
        faceLocCurrentFrame = face_recognition.face_locations(img)
        for y1,x2,y2,x1 in faceLocCurrentFrame:
            face = img[y1:y1+y2, x1:x1+x2, :]
        encode = face_recognition.face_encodings(face)[0]
        encodeList.append(encode)

    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding complete')

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # cv.imshow('face', frame)
    frame = cv.filter2D(src = frame, ddepth = -1, kernel = kernel)
    frame = cv.GaussianBlur(frame, (3,3), cv.BORDER_DEFAULT)
    img = cv.resize(frame, (0,0), None, 0.50, 0.50)
    # img = cv.resize(frame, (800,800), interpolation = cv.INTER_AREA)
    imgCurrentFrame = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    # face_casecade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    # faces = face_casecade.detectMultiScale(imgCurrentFrame)

    # for x,y,w,h in faces:
    #     face = imgCurrentFrame[y:y+h+80, x:x+w, :]
    faceLocCurrentFrame = face_recognition.face_locations(imgCurrentFrame)
    encodeCurrentFrame = face_recognition.face_encodings(imgCurrentFrame, faceLocCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceLocCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)   
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            cv.rectangle(frame, (x1*2, y1*2),(x2*2, y2*2), (0,255,0),2)
            cv.rectangle(frame, (x1*2, y2*2-25),(x2*2, y2*2),(0,255,0), cv.FILLED)
            cv.putText(frame, name, (x1*2 +6, y2*2-2), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
    cv.imshow('face', frame )
    cv.waitKey(1)