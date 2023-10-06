import cv2
from sort import Sort

# Initialize the SORT tracker
tracker = Sort()

# Load the input video
cap = cv2.VideoCapture("input_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform person detection as shown in the previous script
    # Extract bounding box coordinates for detected persons

    # Format the bounding boxes for SORT input
    detections = []
    for (x, y, w, h) in detected_persons:
        detections.append([x, y, x + w, y + h])

    # Update the tracker with the detections
    tracked_objects = tracker.update(np.array(detections))

    # Draw bounding boxes for tracked persons
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Person Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
