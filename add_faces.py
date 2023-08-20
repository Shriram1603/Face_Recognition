import cv2
import pickle
import numpy as np
import os
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


faces_data=[]
i=0
name=input("Enter Your Name")

while True:
    ret,frame=video.read()   #gives 2 values bool if camera running or not and  frame
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        crop_image=frame[y:y+h,x:x+w,:]
        resized_img=cv2.resize(crop_image,(50,50))
        if len(faces_data)<=100 and i%10==0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame,str(len(faces_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1) #face detection
    height, width, _ = frame.shape
    if width > 0 and height > 0:
        cv2.imshow("Frame", frame)
    else:
        print("Invalid frame dimensions")
    k=cv2.waitKey(1)
    if k==ord("q") or len(faces_data)==100:
        break
cv2.destroyAllWindows()
video.release()

#face detection and recognition over

# names storing
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

names_file_path = 'data/names.pkl'

if not os.path.exists(names_file_path):
    names = [name] * 100
else:
    with open(names_file_path, 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100

with open(names_file_path, 'wb') as f:
    pickle.dump(names, f)
        
#storing faces

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl','wb') as f:
        pickle.dump(faces_data,f)
else:
     with open('data/faces_data.pkl','rb') as f:
        faces=pickle.load(f)
     faces=np.append(faces,faces_data,axis=0)
     names=names+[name]*100
     with open('data/faces_data.pkl','wb') as f:
        pickle.dump(faces,f)