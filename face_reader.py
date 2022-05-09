import os
import cv2 as cv
import numpy as np

people = ['Ben', 'dakota','salman' , 'tom' ]
# print(type(people))
dirr = r'D:\Project\face detection\New folder'
haar_cascade = cv.CascadeClassifier('haar_face.xml')
fea = []
labels = []

def create_train():
    for person in people:
        label = people.index(person)
        print(label) 
        path = os.path.join(dirr , person)
        print(path) 
        # 'D:\\Project\\face detection\\New folder\\Ben'
        for img in os.listdir(path):
            # print(img)

            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)

            gray = cv.cvtColor(img_array , cv.COLOR_BGR2GRAY)
                # cv.imshow(' kbcs',faces_rect)
                # cv.waitKey(0)
            faces_rect = haar_cascade.detectMultiScale(gray , 1.1, minNeighbors = 4)
            for ( x ,y, w,h) in faces_rect:
                faces_roi = gray[y:y+h , x: x+w]

                fea.append(faces_roi)

                labels.append(label)

create_train()

fea = np.array(fea , dtype= 'object')
labels = np.array(labels)

face_recogni = cv.face.LBPHFaceRecognizer_create()

face_recogni.train(fea , labels)
face_recogni.save('face_reader.yml')
np.save(' labels.npy', labels)
np.save(' fea.npy', fea)
