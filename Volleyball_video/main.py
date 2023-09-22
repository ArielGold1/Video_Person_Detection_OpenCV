import cv2
import numpy as np

# Load pre-trained Faster R-CNN model
model = cv2.dnn.readNetFromCaffe("model.prototxt", "model.caffemodel")

# Open the video file
cap = cv2.VideoCapture(r'C:\Users\arigp\Downloads\Volleyball.mp4')

# Get the frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the input blob shape for the model
blob = cv2.dnn.blobFromImage(np.zeros((300, 300, 3), dtype=np.uint8), 0.007843, (300, 300), 127.5)

# Set the model to the input blob
model.setInput(blob)

# Loop through each frame of the video
while True:
    # Read the next frame
    ret, frame = cap.read()

    # Break the loop if there are no more frames
    if not ret:
        break

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    model.setInput(blob)

    # Run forward pass through the model
    detections = model.forward()

    # Loop through each detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Get the bounding box coordinates
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Output", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

