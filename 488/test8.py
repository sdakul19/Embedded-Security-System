import cv2
import time

def main():
    # Initialize two cameras
    cap1 = cv2.VideoCapture(0)  # First camera
    cap2 = cv2.VideoCapture(1)  # Second camera
    
    # Check if cameras opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video devices.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    sensitivity_threshold = 5  # Percentage
    face_size_threshold = 10000  # Thresholq for face size to trigger alert and capture

    prev_frame1, prev_frame2 = None, None  # For motion detection
    prev_face_size1, prev_face_size2 = None, None  # For tracking face size changes

    try:
        while True:
            # Read frames from both cameras
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            # If frame read was not successful, break loop
            if not ret1 or not ret2:
                print("Error: Can't receive frame. Exiting ...")
                break

            # Process each frame
            for idx, (frame, cap, prev_frame, prev_face_size) in enumerate(zip([frame1, frame2], [cap1, cap2], [prev_frame1, prev_frame2], [prev_face_size1, prev_face_size2]), 1):
                processed_frame, new_prev_frame, new_prev_face_size = process_frame(frame, face_cascade, prev_frame, prev_face_size, sensitivity_threshold, face_size_threshold)
                if idx == 1:
                    prev_frame1, prev_face_size1 = new_prev_frame, new_prev_face_size
                else:
                    prev_frame2, prev_face_size2 = new_prev_frame, new_prev_face_size
                
                # Display the processed frame
                cv2.imshow(f'Camera {idx}', processed_frame)

            # If 'q' is pressed, break from the loop
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # Release resources and destroy all windows
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

def process_frame(frame, face_cascade, prev_frame, prev_face_size, sensitivity_threshold, face_size_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        return frame, gray_blurred, prev_face_size

    frame_diff = cv2.absdiff(prev_frame, gray_blurred)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = len(contours) > 0

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(60, 60))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            current_face_size = w * h

            # Draw a blue rectangle around every detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Additional check for the largest face if it's approaching
            if current_face_size > face_size_threshold and prev_face_size is not None:
                percentage_increase = ((current_face_size - prev_face_size) / prev_face_size) * 100
                if percentage_increase > sensitivity_threshold:
                    # Draw a red rectangle and show alert for the approaching largest face
                    cv2.putText(frame, 'ALERT: Suspicious Person Approaching', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"approaching_face_{timestamp}.jpg"
                    cv2.imwrite('/Users/nabeehdaouk/source/repos/GitHub/Embedded-Security-System/488/suspects/'+ filename, frame)
                    time.sleep(.1)

            # Update previous face size only if the current face is the largest detected face
            if current_face_size == max([w*h for (x, y, w, h) in faces]):
                prev_face_size = current_face_size
    else:
        prev_face_size = None

    if motion_detected:
        cv2.putText(frame, "Motion Detected", (1600, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Motion Detected", (1600, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    return frame, gray_blurred, prev_face_size

if __name__ == '__main__':
    main()

