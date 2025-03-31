import cv2
import torch
import threading
from flask import Flask, render_template, jsonify
from ultralytics import YOLO
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# RTSP stream URL
rtsp_url = "rtsp://100.69.226.196:554"

# Open RTSP stream
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Failed to connect to RTSP stream. Check the URL and network.")
    exit()

# Global variables for threading
frame = None
lock = threading.Lock()

# Capture frames in a separate thread for better performance
def capture_frames():
    global frame
    while cap.isOpened():
        ret, new_frame = cap.read()
        if ret:
            with lock:
                frame = new_frame

# Start the capture thread
threading.Thread(target=capture_frames, daemon=True).start()

# Serve the HTML page when visiting the root
@app.route('/')
def index():
    return render_template('index.html')  # You can change this if your HTML file is named differently

# Route for serving the data to the website
@app.route('/data')
def data():
    with lock:
        if frame is None:
            return jsonify({'people_count': 0, 'wait_time': 0})

        current_frame = frame.copy()

    # Resize for faster processing
    resized_frame = cv2.resize(current_frame, (640, 360))

    # Run YOLO model
    results = model(resized_frame)

    # Person count
    person_count = 0

    # Draw bounding boxes for detected people
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class ID
            conf = box.conf[0].item()   # Confidence score

            if class_id == 0 and conf > 0.5:  # Class ID 0 is "person"
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Estimate wait time (5 sec per person)
    wait_time = person_count * 5

    # Display the person count and wait time on the local feed (laptop)
    cv2.putText(resized_frame, f"People: {person_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(resized_frame, f"Wait Time: {wait_time} sec", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with bounding boxes on your laptop screen
    cv2.imshow("Real-Time Dining Hall Feed", resized_frame)

    # Return the people count and wait time to the website
    return jsonify({'people_count': person_count, 'wait_time': wait_time})

# Main block to run Flask app
if __name__ == '__main__':
    # Run Flask without reloader to avoid issues with threading
    app.run(debug=True, host='0.0.0.0', port=8000, use_reloader=False)

    # Show the camera feed on your laptop
    while True:
        # You can show the frame on your laptop here, separate from the Flask app
        with lock:
            if frame is not None:
                cv2.imshow("Real-Time Dining Hall Feed", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

