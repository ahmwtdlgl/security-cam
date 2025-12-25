from pushbullet import Pushbullet
import cv2
import mediapipe as mp
import time

API_KEY = "o.0Mw62qny6PvcFIUDzUumtU1uBnKZJv5U"  # kendi anahtarın
pb = Pushbullet(API_KEY)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

last_push_time = 0  # aynı kişiyi sürekli göndermemek için zaman kontrolü

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Vücut noktalarını çiz
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
            )

            # 🔹 Vücut algılandı -> sadece 10 saniyede bir bildirim gönder
            if time.time() - last_push_time > 10:
                cv2.imwrite("insan.jpg", image)
                print("✅ Vücut algılandı, fotoğraf kaydedildi!")

                # Fotoğrafı yükle ve gönder
                with open("insan.jpg", "rb") as f:
                    file_data = pb.upload_file(f, "insan.jpg")
                pb.push_file(**file_data, body="Kamerada bir kişi algılandı 👀")

                last_push_time = time.time()

        else:
            cv2.putText(image, "Vucut algilanmadi", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC çıkış
            break

cap.release()
cv2.destroyAllWindows()
