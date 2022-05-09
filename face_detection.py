from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
import cv2 as cv

people = ['Ben', 'dakota','salman' ,'tom']
haar_cascade = cv.CascadeClassifier('haar_face.xml') 
        
face_recogni = cv.face.LBPHFaceRecognizer_create()
face_recogni.read('face_detction.yml')

img = cv.imread(r'D:\\Project\\face detection\\New folder\\dakota\\download2.jfif')
# img = cv.imread(r'D:\\Project\\face detection\\New folder\\Ben\\download.jfif')
# img = cv.imread(r'D:\\Project\\face detection\\New folder\\tom\\download.jfif')
# img = cv.imread(r'D:\\Project\\face detection\\New folder\\salman\\download1.jfif')

gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('' , gray)

faces_rect = haar_cascade.detectMultiScale(gray ,1.1 ,7)

for ( x, y, w, h ) in faces_rect:
    faces_roi = 4*gray [ y:y+h , x : x + w]
    print(faces_roi)
    label, confidence = face_recogni.predict(faces_roi/4)
    print(f'label = {people [ label]} with  a confidence= { confidence}')
    cv.putText(img ,str(people [ label]) , ( 20 ,20 ) ,cv.FONT_HERSHEY_COMPLEX,1.0 ,(0 ,255,0) , thickness= 2)
    cv.rectangle(img , (x ,y) , (x +w ,y+h) , (0 ,255 ,0) , 2)


cv.imshow('', img )
cv.waitKey(0)