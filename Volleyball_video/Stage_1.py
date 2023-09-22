from curses import COLORS
import numpy as np
import LABELS as LABELS
import cv2

from PlayerTracking import PlayerTracking

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Create a VideoCapture object to capture the video stream
cap = cv2.VideoCapture("r'C:\Users\arigp\Downloads\Volleyball.mp4'")

while True:
    # Capture each frame of the video
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Preprocess the frame
    player_tracker = PlayerTracking()
    frame = player_tracker(frame)

    # Get the dimensions of the frame
    height, width = frame.shape[:2]

    # Construct a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Pass the blob through the network
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialize the bounding boxes, confidences and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression to the bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


