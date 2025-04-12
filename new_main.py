import cv2
import face_recognition
import mediapipe as mp

#media tools
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
#starting lists
known_face_encodings=[]
known_face_names=[]

#known person faces& names

# List of (name, image path) tuples
known_faces = [
    ("Bobby", "known/bobby1.jpg"),
    ("Chara", "known/bobby2.jpg"),
    ("Modi", "known/modi2.jpg"),
    ("Manu", "known/small_manu.jpg"),
    ("Mangona", "known/big_manu.jpg"),
    ("Mohan-sir", "known/mohan_sir.jpg"),
    ("Rohini-sir", "known/Rohinisir.jpg"),
    ("Koushik", "known/koushik.jpg")
]

# Load and encode each known face
for name, path in known_faces:
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)








#webcam
video_capture=cv2.VideoCapture(0)

while True:
    ret,frame=video_capture.read()

    face_loc = face_recognition.face_locations(frame)
    face_encode = face_recognition.face_encodings(frame, face_loc)








    # Scale face location back to original size
    for (top, right, bottom, left), face_encode in zip(face_loc, face_encode):
        matches = face_recognition.compare_faces(known_face_encodings,face_encode)
        name="unknown"
        


        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame,(left,top),(right,bottom),(195,186,283),2)
        cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_DUPLEX,1,(195,186,283),2)

    image = cv2.flip(cv2.imread(frame), 1)

    cv2.imshow("Video",image)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

#------------------------------------------------------------------------------------------------------------------------------


