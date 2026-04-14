import cv2

# load haarcascade file
face_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

# start webcam
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Face Detection Video", frame)

    if cv2.waitKey(1) == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()