import cv2

def main():
    # Initialize the video capture object. 0 is usually the default camera.
    cap = cv2.VideoCapture(0)

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Play the live video feed
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame is read correctly, ret is True
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Convert the frame to grayscale for the face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            # Check if any faces are detected
            if len(faces) > 0:
                # Draw rectangles around the faces and display text
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            else:
                # Optional: Display text when no faces are detected
                cv2.putText(frame, 'No Face Detected', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Display the resulting frame with detected faces
            cv2.imshow('Live Camera Feed with Face Detection', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
