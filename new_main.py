import cv2
import face_recognition
import mediapipe as mp

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

# Lists for known face data
known_face_encodings = []
known_face_names = []

# List of known faces (Name, Image Path)
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

# Load and encode known faces
for name, path in known_faces:
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
    else:
        print(f"⚠️ No face found in {path}")

# Start video capture
video_capture = cv2.VideoCapture(0)

with mp_hand.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Flip the frame once and use throughout
        frame = cv2.flip(frame, 1)

        # --- FACE RECOGNITION ---
        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]

            # Draw face rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        # --- HAND GESTURE DETECTION ---
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        hand_results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)

        # Show final output
        cv2.imshow("Face Recognition + Hand Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
