import cv2 as cv
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
import face_recognition
# cap = cv.VideoCapture(0)

imgDav = face_recognition.load_image_file('images/Aruna.jpg')
imgDav = cv.cvtColor(imgDav, cv.COLOR_BGR2RGB)
imgDav = cv.resize(imgDav, (0,0), None, 0.50, 0.50)
# imgDav = cv.resize(imgDav, (740,800), interpolation = cv.INTER_AREA)
cv.imshow("frame", imgDav)
# imgDav_test = face_recognition.load_image_file('images/Isuri.jpg')
# imgDav_test = cv.cvtColor(imgDav_test, cv.COLOR_BGR2RGB)
# imgDav_test = cv.resize(imgDav_test, (500,500), interpolation = cv.INTER_AREA)
# faceLoc = face_recognition.face_locations(imgDav)[0]
# encodeDav = face_recognition.face_encodings(imgDav)[0]
# faceLoc_test = face_recognition.face_locations(imgDav_test)[0]
# encodeDav_test = face_recognition.face_encodings(imgDav_test)[0]
# print(faceLoc)
# face_casecade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# test1 = cv.imread("images/Isuri.jpg")
# test1 = cv.resize(test1, (0,0), None, 0.50, 0.50)
# faces1 = face_casecade.detectMultiScale(test1)
# for x,y,w,h in faces1:
#     face = test1[y:y+h, x:x+w, :]

# cv.imshow("face",face)
# cv.waitKey(0)

# test2 = cv.imread("images/test.jpg")
# # test2 = cv.resize(test2, (0,0), None, 0.25, 0.25)
# faces2 = face_casecade.detectMultiScale(test2)
# for x,y,w,h in faces2:
#     face = test2[y:y+h, x:x+w, :]
#     encodeDav_test2 = face_recognition.face_encodings(face)[0]

# matches = face_recognition.compare_faces(encodeDav_test1, encodeDav_test2)
# faceDis = face_recognition.face_distance(encodeDav_test1, encodeDav_test2) 
# print(faceDis)
# test1 = cv.imread("images/Isuri.jpg")
# test1 = cv.resize(test1, (0,0), None, 0.25, 0.25)
# test = cv.imread("images/test.jpg")
# test = cv.resize(test, (0,0), None, 0.25, 0.25)
# faceLoc_test1 = face_recognition.face_locations(test1)[0]
# encodeDav_test1 = face_recognition.face_encodings(test1)[0]

# faceLoc_test2 = face_recognition.face_locations(test)[0]
# encodeDav_test2 = face_recognition.face_encodings(test)[0]

# matches = face_recognition.compare_faces(encodeDav_test1, encodeDav_test2)
# faceDis = face_recognition.face_distance(encodeDav_test1, encodeDav_test2) 
# print(faceDis)
# results = face_recognition.compare_faces([encodeDav],encodeDav_test)
# # distance face
# faceDis = face_recognition.face_distance([encodeDav],encodeDav_test)
# cv.putText(imgDav_test,f'{results} {round(faceDis[0],2)}', (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
# cv.imshow("imgDav_test", imgDav)
# cv.imshow("imgDav_test", imgDav_test)
cv.waitKey(0)