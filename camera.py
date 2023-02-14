import cv2
import os
import datetime

# 카메라 모듈 초기화
camera = cv2.VideoCapture(0)

# life 폴더가 없다면 생성
current_dir = os.path.abspath(os.path.dirname(__file__))
life_folder = os.path.join(current_dir, "life")
if not os.path.exists(life_folder):
    os.makedirs(life_folder)
print("Absolute path of life folder:", life_folder)  
# 카메라로부터 얼굴 탐지
while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(r"C:\Python27\haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join('life', 'face_{}.jpg'.format(now)), face)
        camera.release()
        cv2.destroyAllWindows()
        exit()

    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
