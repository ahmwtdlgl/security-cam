from pushbullet import Pushbullet
import cv2
import mediapipe as mp
import time

API_KEY = "your api key"  #your api key enter there
pb = Pushbullet(API_KEY)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

last_push_time = 0  

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

       
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
            )

          
            if time.time() - last_push_time > 10:
                cv2.imwrite("insan.jpg", image)
                print(" Body detected , saving image!")

              
                with open("insan.jpg", "rb") as f:
                    file_data = pb.upload_file(f, "insan.jpg")
                pb.push_file(**file_data, body="someone detected ")

                last_push_time = time.time()

        else:
            cv2.putText(image, "couldnt recognize a person", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  
            break

cap.release()
cv2.destroyAllWindows()

