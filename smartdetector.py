import cv2

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray,1.1,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        if len(faces)>0:
            cv2.putText(frame, "face detected", (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        reg_gray=gray[y:y+h , x:x+w]
        reg_color=frame[y:y+h , x:x+w]

        eyes=eye_cascade.detectMultiScale(reg_gray,1.5,10)

        if len(eyes)>0:

          cv2.putText(frame,"eyes detected",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        smile=smile_cascade.detectMultiScale(reg_gray,1.7,20)

        if len(smile)>0:
          cv2.putText(frame, "smiling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("smartdetector",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyWindow()
