import cv2
import time
from tkinter import *
from threading import Thread
import numpy as np

def video_stream():
    global cap, frame, is_running, frame_lock
    while is_running:
        ret, temp_frame = cap.read()
        if not ret:
            break
        with frame_lock:
            frame = temp_frame.copy()
        detect_faces_and_motion(frame)
        cv2.imshow('Live Camera Feed with Face and Motion Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

def capture_image():
    global frame, frame_lock
    with frame_lock:
        if frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"captured_image_{timestamp}.jpg"
            cv2.imwrite('/path/to/save/' + filename, frame)  # Make sure to use an actual path
            print(f"Image saved as {filename}")
        else:
            print("No frame to capture.")

def stop_system():
    global is_running
    is_running = False
    cap.release()
    cv2.destroyAllWindows()
    window.quit()

def create_gui():
    global window
    window = Tk()
    window.title("Security System Control Panel")
    window.geometry('300x150')

    start_button = Button(window, text="Start", command=lambda: Thread(target=video_stream).start())
    start_button.pack(pady=10)

    capture_button = Button(window, text="Capture Image", command=capture_image)
    capture_button.pack(pady=10)

    stop_button = Button(window, text="Stop & Quit", command=stop_system)
    stop_button.pack(pady=10)

    window.mainloop()

# Global variables
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
prev_face_size = None
sensitivity_threshold = 5
prev_frame = None
is_running = True
frame = None
frame_lock = threading.Lock()

if __name__ == '__main__':
    create_gui()
