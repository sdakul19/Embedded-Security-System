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
                processed_frame, new_prev_frame, new_prev_face_size = process_frame(frame, face_cascade, prev_frame, prev_face_size, sensitivity_threshold)
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

def process_frame(frame, face_cascade, prev_frame, prev_face_size, sensitivity_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        return frame, gray_blurred, prev_face_size

    frame_diff = cv2.absdiff(prev_frame, gray_blurred)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = len(contours) > 0

    # Adjust the scaleFactor to a higher value, increase minNeighbors, and specify a larger minSize
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(60, 60))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0:1]
        largest_face = faces[0]
        x, y, w, h = largest_face
        current_face_size = w*h
        rectangle_color = (255, 0, 0)

        if prev_face_size is not None:
            percentage_increase = ((current_face_size - prev_face_size) / prev_face_size) * 100
            if percentage_increase > sensitivity_threshold:
                rectangle_color = (0, 0, 255)
                cv2.putText(frame, 'ALERT: Face Approaching', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, rectangle_color, 3)

        cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)
        prev_face_size = current_face_size
    else:
        prev_face_size = None

    if motion_detected:
        cv2.putText(frame, "Motion Detected", (1600, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Motion Detected", (1600, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    return frame, gray_blurred, prev_face_size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)  # Blur to reduce noise

    if prev_frame is None:
        return frame, gray_blurred, prev_face_size

    # Compute difference between current frame and previous frame for motion detection
    frame_diff = cv2.absdiff(prev_frame, gray_blurred)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = len(contours) > 0

    # Face detection logic - only detecting the largest face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        # Sort faces based on area (w*h) and take the largest one
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0:1]
        largest_face = faces[0]
        x, y, w, h = largest_face
        current_face_size = w*h

        # Default rectangle color to blue
        rectangle_color = (255, 0, 0)

        if prev_face_size is not None:
            percentage_increase = ((current_face_size - prev_face_size) / prev_face_size) * 100
            if percentage_increase > sensitivity_threshold:
                # Change rectangle color to red if face is approaching
                rectangle_color = (0, 0, 255)
                cv2.putText(frame, 'ALERT: Face Approaching', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, rectangle_color, 3)

        # Draw the rectangle around the largest face with the determined color
        cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)

        prev_face_size = current_face_size
    else:
        prev_face_size = None

    if motion_detected:
        cv2.putText(frame, "Motion Detected", (1600, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Motion Detected", (1600, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    return frame, gray_blurred, prev_face_size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)  # Blur to reduce noise

    if prev_frame is None:
        return frame, gray_blurred, prev_face_size

    # Compute difference between current frame and previous frame for motion detection
    frame_diff = cv2.absdiff(prev_frame, gray_blurred)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = len(contours) > 0

    # Face detection logic
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        largest_face = faces[0]
        x, y, w, h = largest_face
        current_face_size = w*h
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if prev_face_size is not None:
            percentage_increase = ((current_face_size - prev_face_size) / prev_face_size) * 100
            if percentage_increase > sensitivity_threshold:
                cv2.putText(frame, 'ALERT: Face Approaching', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"approaching_face_{timestamp}.jpg"
                cv2.imwrite('/Users/nabeehdaouk/source/repos/GitHub/Embedded-Security-System/488/suspects/'+ filename, frame)
                time.sleep(.1)
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