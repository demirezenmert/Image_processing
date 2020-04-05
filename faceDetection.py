import cv2 
import imageio


face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')

def detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face2 = faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame



if __name__ == "__main__":
    print('***'*10)
    print('''
                 Created by Mert Demirezen 
                 Copyright © 2019 Mert Demirezen. All rights reserved.


                Welcome Face And Eyes detection on picture program
                Choose Picture: Ensure picture in the same 
                directory with 'faceDetection.py'
                or enter with picture via path
    
    ''')
    print('***'*10)
    user_image = input('Please Enter picture name with extensions:\n')
    image = imageio.imread(user_image)
    image = detection(frame=image)
    imageio.imwrite(user_image+'Detected.jpg',image)
    print('....Succes.....')