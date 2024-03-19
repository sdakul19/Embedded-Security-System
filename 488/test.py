import cv2

def main():
    # Initialize the video capture object. 0 is usually the default camera.
    cap = cv2.VideoCapture(0)

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

            # Display the resulting frame
            cv2.imshow('Live Camera Feed', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
