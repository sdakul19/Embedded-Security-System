import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    prev_face_size = None
    sensitivity_threshold = 5  # Percentage
    prev_frame = None  # For motion detection

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)  # Blur to reduce noise

            if prev_frame is None:
                prev_frame = gray_blurred
                continue

            # Compute difference between current frame and previous frame
            frame_diff = cv2.absdiff(prev_frame, gray_blurred)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = len(contours) > 0

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                largest_face = faces[0]
                x, y, w, h = largest_face
                current_face_size = w*h
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                if prev_face_size:
                    percentage_increase = ((current_face_size - prev_face_size) / prev_face_size) * 100
                    if percentage_increase > sensitivity_threshold:
                        cv2.putText(frame, 'ALERT: Face Approaching', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"approaching_face_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        time.sleep(.1)
                prev_face_size = current_face_size
            else:
                prev_face_size = None

            if motion_detected:
                cv2.putText(frame, "Motion Detected", (1600, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Motion Detected", (1600, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.imshow('Live Camera Feed with Face and Motion Detection', frame)
            prev_frame = gray_blurred

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
