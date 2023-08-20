from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
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



while True:
    ret,frame=video.read()   #gives 2 values bool if camera running or not and  frame
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        crop_image=frame[y:y+h,x:x+w, :]
        resized_img = cv2.resize(crop_image,(50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1) #face detection
    height, width, _ = frame.shape
    if width > 0 and height > 0:
        cv2.imshow("Frame", frame)
    else:
        print("Invalid frame dimensions")
    k=cv2.waitKey(1)
    if k==ord("q"):
        break
cv2.destroyAllWindows()
video.release()
