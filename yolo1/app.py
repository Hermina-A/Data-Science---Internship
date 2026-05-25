import cv2
from ultralytics import YOLO

# 1. Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# 2. Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' on the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Get all predictions from YOLO
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    
    # This will hold our single best physical object
    best_object_box = None

    # YOLO automatically sorts objects from highest to lowest confidence.
    # We check each one and pick the first one that is NOT a person (Class ID 0).
    for box in boxes:
        class_id = int(box.cls[0])
        
        if class_id != 0:  # If the detected item is NOT a person, grab it!
            best_object_box = box
            break  # Stop immediately so we only handle this ONE object

    # If a physical object is found, draw ONLY that one box
    if best_object_box is not None:
        x1, y1, x2, y2 = map(int, best_object_box.xyxy[0])
        conf = float(best_object_box.conf[0])
        class_id = int(best_object_box.cls[0])
        label_name = model.names[class_id] # Convert ID to name (e.g., "cell phone")
        
        # Draw a green box around it
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add the label text
        label = f"{label_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the live window
    cv2.imshow("YOLOv8 Strict Single Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()