import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video= cv2.VideoCapture(0)

# load known faces

my= face_recognition.load_image_file("dhruvin.jpg")
my_encoded= face_recognition.face_encodings(my)[0]
kisu= face_recognition.load_image_file("krishna.jpeg")
kisu_encoded= face_recognition.face_encodings(kisu)[0]
manjula= face_recognition.load_image_file("mummy.jpeg")
manjula_encoded= face_recognition.face_encodings(manjula)[0]
pichai= face_recognition.load_image_file("pichai.jpg")
pichai_encoded= face_recognition.face_encodings(pichai)[0]
 
 
known_face_encoding= [my_encoded,kisu_encoded,manjula_encoded,pichai_encoded]
known_face_names= ["Dhruvin","kisu","manjula","pichai"]

# list of expected participeints

participeints=known_face_names.copy()

# get time and date fot attendence

now=datetime.now()
date= now.strftime("%d-%m-%Y")

f =open(f"attendece of {date}.csv", "w+", newline="")
lnwriter= csv.writer(f)

while True:
    _,frame=video.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame= cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
    
    # recognixing face
    face_locations= face_recognition.face_locations(rgb_small_frame)
    face_encodings= face_recognition.face_encodings(rgb_small_frame,face_locations)
    
    for face_encoding in face_encodings:
        matches= face_recognition.compare_faces(known_face_encoding,face_encoding)
        face_distance= face_recognition.face_distance(known_face_encoding,face_encoding)
        best_match_index= np.argmin(face_distance)

        if(matches[best_match_index]):
            name= known_face_names[best_match_index]
        
            # adding name of the person detected
            if name in known_face_names:
                font =cv2.FONT_HERSHEY_SIMPLEX
                bottonLeft=(10,100)
                fontscale= 1.5
                fontcolor=(57,255,20)
                thickness=2
                linetype=2
                cv2.putText(frame, name + " present ", bottonLeft,font,fontscale,fontcolor,thickness,linetype)

                if name in participeints:
                    participeints.remove(name)
                    current_time= now.strftime("%H:%M:%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("Align your face properly with camera, press q to quit",frame)
    if cv2.waitKey(1) & 0xFF== ord("q"):
        break
    
video.release()
cv2.destroyAllWindows()
f.close()