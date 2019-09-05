# By Luiz Viana
import cv2
import os
import numpy as np

# FUNCTIONS
def savePerson():
    global lastName
    global boolsaveimg
    print('Qual seu nome?')
    name = input()
    lastName = name
    boolsaveimg = True

def trainData():
    global recognizer
    global trained
    global persons
    trained = True
    persons = os.listdir('train')

    ids = []
    faces = []
    for i,p in enumerate(persons):
        for f in os.listdir(f'train/{p}'):
            img = cv2.imread(f'train/{p}/{f}',0)
            faces.append(img)
            ids.append(i)
    recognizer.train(faces, np.array(ids))


def saveImg(img):
    global lastName
    if not os.path.exists('train'):
        os.makedirs('train')

    if not os.path.exists(f'train/{lastName}'):
        os.makedirs(f'train/{lastName}')

    files = os.listdir(f'train/{lastName}')
    cv2.imwrite(f'train/{lastName}/{str(len(files))}.jpg',img)


lastName = ""
boolsaveimg = False
trained = False
savecount = 0
persons = []

recognizer = cv2.face.LBPHFaceRecognizer_create()

#CAPTURE WEBCAM
cap = cv2.VideoCapture(0)

#CAPTURE VIDEO FILE
#cap = cv2.VideoCapture("video.mp4")

# LOAD HAAR CASCADE
face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')

while(True):
    ret, frame = cap.read()
    normalframe = cv2.flip(frame, +1)
    # GRAY FRAME
    gray = cv2.cvtColor(normalframe, cv2.COLOR_RGB2GRAY)

    # DETECT FACES IN FRAME
    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    # ALL FOUND FACES
    for (x,y,w,h) in faces:
        # FACE'S ROI
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi,(50,50))
        # DRAW RECTANGLE IN FRAME
        cv2.rectangle(normalframe, (x,y), (x+w,y+h), (255,0,0),3)
        if trained:
            idf, conf = recognizer.predict(roi)
            nameP = persons[idf]
            cv2.putText(normalframe,nameP, (x, y), 0, 0.5,(0,255,0),1, cv2.LINE_AA)

        if boolsaveimg:
            saveImg(roi)
            savecount += 1

        if(savecount > 50):
            boolsaveimg = False
            savecount = 0

    cv2.imshow('normalframe', normalframe)
    #cv2.imshow('gray', gray)

    key = cv2.waitKey(1)

    # PRESS "s" TO SALVE FACE IMAGES
    if key == ord('s'):
        savePerson()

    # PRESS "t" TO TRAIN IMAGE
    if key == ord('t'):
        trainData()

    # PRESS "q" TO QUIT 
    if key == ord('q'):
        break

# CACHE FLUSH
cap.release()

# DESTROY ALL WINDOWS
cv2.destroyAllWindows()
