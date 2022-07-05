import cv2 as cv
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
import face_recognition
# cap = cv.VideoCapture(0)

imgDav = face_recognition.load_image_file('images/dav3.jpg')
imgDav = cv.cvtColor(imgDav, cv.COLOR_BGR2RGB)
imgDav = cv.resize(imgDav, (500,500), interpolation = cv.INTER_AREA)
imgDav_test = face_recognition.load_image_file('images/is.jpg')
imgDav_test = cv.cvtColor(imgDav_test, cv.COLOR_BGR2RGB)
imgDav_test = cv.resize(imgDav_test, (500,500), interpolation = cv.INTER_AREA)
faceLoc = face_recognition.face_locations(imgDav)[0]
encodeDav = face_recognition.face_encodings(imgDav)[0]
faceLoc_test = face_recognition.face_locations(imgDav_test)[0]
encodeDav_test = face_recognition.face_encodings(imgDav_test)[0]

results = face_recognition.compare_faces([encodeDav],encodeDav_test)
# distance face
faceDis = face_recognition.face_distance([encodeDav],encodeDav_test)
cv.putText(imgDav_test,f'{results} {round(faceDis[0],2)}', (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
cv.imshow("imgDav_test", imgDav)
cv.imshow("imgDav_test", imgDav_test)
cv.waitKey(0)