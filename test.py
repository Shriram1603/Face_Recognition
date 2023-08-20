from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


with open('data/names.pkl','rb') as w:
    LABELS=pickle.load(w)
    LABELS = np.array(LABELS)
with open('data/faces_data.pkl','rb') as f:
    FACES=pickle.load(f)

print("Faces shape:", FACES.shape)
print("Labels shape:", LABELS.shape)


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)

imgBackground=cv2.imread("back1.jpg")

COL_NAMES=['NAME','TIME','DATE']

attendance=[]

date = datetime.now().strftime("%d-%m-%Y")  
while True:
    frame_attendance = []
    ret,frame=video.read()   #gives 2 values bool if camera running or not and  frame
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        crop_image=frame[y:y+h,x:x+w, :]
        resized_img = cv2.resize(crop_image,(50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.now().strftime("%d-%m-%Y")
        timestamp=datetime.now().strftime("%H:%M-%S")
        exist=os.path.isfile("Attendance/Attendance_"+ date +".csv")
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1) #face detection
        frame_attendance.append([str(output[0]), str(timestamp),str(date)])

    # Append the data for all detected faces in this frame to attendance
    # attendance.extend(frame_attendance)
    imgBackground[150:150 + frame.shape[0], 55:55 + frame.shape[1]] = frame
    height, width, _ = frame.shape
    if width > 0 and height > 0:
        cv2.imshow("Frame", imgBackground)
    else:
        print("Invalid frame dimensions")
    k=cv2.waitKey(1)
    if k==ord('o'):
        attendance.extend(frame_attendance)

        if exist:
            with open("Attendance/Attendance_"+date+".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
                csvfile.close()
            attendance=[]
        else:
            with open("Attendance/Attendance_"+date+".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
        attendance=[]
        
        
    if k==ord("q"):
        break
cv2.destroyAllWindows()
video.release()
