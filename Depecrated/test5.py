import cv2
import time

def main():
    # Initialize the video capture object. 0 is usually the default camera.
    cap = cv2.VideoCapture(0)

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Previous face size (initially None)
    prev_face_size = None

    # Sensitivity threshold for detecting approach (percentage increase in face size)
    sensitivity_threshold = 5  # Percentage

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
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Sort faces based on the area (w * h) in descending order and select the largest
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                largest_face = faces[0]
                x, y, w, h = largest_face
                current_face_size = w*h

                # Draw a rectangle around the largest face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Check if the largest face is approaching
                if prev_face_size:
                    percentage_increase = ((current_face_size - prev_face_size) / prev_face_size) * 100
                    if percentage_increase > sensitivity_threshold:
                        cv2.putText(frame, 'ALERT: Face Approaching', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

                        # Generate a timestamp
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"approaching_face_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        time.sleep(.1)
                # Update previous face size
                prev_face_size = current_face_size
            else:
                # Reset previous face size if no faces are detected
                prev_face_size = None

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
