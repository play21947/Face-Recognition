import cv2 as cv
import cv2

video = cv.VideoCapture('./two5.mp4')



haar = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

clf = cv.face.LBPHFaceRecognizer_create()
clf.read('./classifier.xml')


id_c = 0

id_define = 1

coords = []

while True:
    _, frame = video.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = haar.detectMultiScale(gray)

    for (x,y,w,h) in faces:


        id,con = clf.predict(gray[y:y+h, x:x+w])

        confidence = round(100 - con)

        print("confidence : ", confidence,"%")

        if(confidence > 40):
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv.putText(frame, "Play2", (x, y-4), cv.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
        else:
            if(confidence > 0):
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv.putText(frame, "Person", (x, y-4), cv.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)


        # print(str(con))

        coords = [x, y, w, h]

    # Add Image to Data for train model
    
    # if(len(coords) == 4):
    #     if(confidence > 0):
    #         result = frame[coords[1]:coords[1]+coords[3], coords[0]: coords[0]+coords[2]]
    #         cv.imwrite('./data/'+'pic.'+str(id_define)+'.'+str(id_c)+'.jpg', result)
    #         id_c = id_c + 1
    #     else:
    #         print("less")

    cv.imshow('main', frame)
    cv.waitKey(30)
