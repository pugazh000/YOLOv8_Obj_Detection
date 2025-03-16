from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("Your_path/runs/detect/train11/weights/best.pt")

# Load test image
img_path = "Your_path/YOLO/test.jpg" ## uses the image which should be downloaded
results = model(img_path)

# Show results
cv2.imshow("Detection", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()

######################  or  ##################################

from ultralytics import YOLO
import cv2

# Load the trained YOLO model (make sure to provide the correct path to your 'best.pt' model file)
model = YOLO("Your_path/runs/detect/train11/weights/best.pt")  # Change the path if necessary

# Initialize the webcam (0 is the default camera, change if you have multiple cameras)
cap = cv2.VideoCapture(0)  # use your realtime webcam to capture

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference on the frame using YOLO model
    results = model(frame)

    # Render the results on the frame
    frame_with_results = results[0].plot()

    # Show the frame with detections
    cv2.imshow("Object Detection", frame_with_results)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
