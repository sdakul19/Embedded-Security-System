import cv2
import time

def main():
    # Initialize the video capture object. 0 is usually the default camera.
    cap = cv2.VideoCapture(0)

    # Load the pre-trained Haar Cascade classifiers for face and full body detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')  # Full body detector

    # Previous sizes (initially empty)
    prev_sizes = []

    # Sensitivity threshold for detecting approach (percentage increase in size)
    sensitivity_threshold = 5  # Percentage

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Convert the frame to grayscale for the detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect full bodies in the image
            bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Current sizes
            current_sizes = [w*h for (x, y, w, h) in bodies]

            # Compare current sizes with previous to detect if a human figure is approaching
            if prev_sizes:
                for size, prev_size in zip(current_sizes, prev_sizes):
                    percentage_increase = ((size - prev_size) / prev_size) * 100
                    if percentage_increase > sensitivity_threshold:
                        cv2.putText(frame, 'ALERT: Human Approaching', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        # Generate a timestamp
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = "human_approaching_" + timestamp + ".jpg"
                        cv2.imwrite(filename, frame)
                        break

            # Update previous sizes
            prev_sizes = current_sizes

            # Draw rectangles around the full bodies
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Live Camera Feed with Human Detection', frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
