import cv2
import os
import time
import json
from datetime import datetime
from ultralytics import YOLO
from ocr_utilsvishallaptop import read_plate

# ==========================================
# MAIN LIVE VIDEO & INDIVIDUAL BATCH LOGGING
# ==========================================
def run_live_feed(source=0, camera_id="123"):  # Set to "123", "124", or "125" as needed
    print("Loading License Plate YOLO model (Intel OpenVINO Accelerated)...")
    yolo_model = YOLO('license_plate_model_openvino_model') 
    
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    last_detections = [] 
    
    last_count_time = time.time()
    unique_plates_in_window = set() 

    while cap.isOpened():
        # Grab frame directly to eliminate lag buffer
        ret = cap.grab()
        if not ret: 
            break
        ret, frame = cap.retrieve()
        if not ret: 
            break
            
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        
        # Inference every 5th frame
        if frame_count % 5 == 0: 
            results = yolo_model(frame, verbose=False)
            last_detections.clear()
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Bounding box padding
                        h, w = frame.shape[:2]
                        px, py = int((x2 - x1) * 0.05), int((y2 - y1) * 0.10)
                        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
                        cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
                        if cx2 <= cx1 or cy2 <= cy1: 
                            continue

                        cropped = frame[cy1:cy2, cx1:cx2]
                        plate_text, confidence = read_plate(cropped)

                        if plate_text and plate_text != "UNKNOWN":
                            print(f"[SYSTEM LOG] Detected Plate: {plate_text}")
                            unique_plates_in_window.add(plate_text)
                            
                        last_detections.append({
                            "box": (x1, y1, x2, y2),
                            "text": plate_text if plate_text else ""
                        })
                        
        # 1. Draw persistent bounding boxes & text
        for det in last_detections:
            bx1, by1, bx2, by2 = det["box"]
            text = det["text"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            if text:
                cv2.putText(frame, text, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 2. 15-Second Individual Export Flush
        current_timer = time.time()
        if current_timer - last_count_time >= 15.0:
            if len(unique_plates_in_window) > 0:
                # Format to match MongoDB ISO standard exactly
                current_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                
                os.makedirs("../database", exist_ok=True)
                # Point to log.json directly
                batch_json_path = os.path.join("..", "database", "log.json")
                
                if os.path.exists(batch_json_path):
                    try:
                        with open(batch_json_path, "r") as file:
                            all_batches = json.load(file)
                    except json.JSONDecodeError:
                        all_batches = []
                else:
                    all_batches = []

                # Format dictionary exactly as Vercel expects
                for plate in unique_plates_in_window:
                    all_batches.append({
                        "time_stamp": current_time,
                        "cam_id": camera_id,
                        "vehicle_id": plate
                    })

                with open(batch_json_path, "w") as file:
                    json.dump(all_batches, file, indent=4)
                    
                print(f"[BATCH SENT] Logged {len(unique_plates_in_window)} individual records to log.json at {current_time}.")
            
            unique_plates_in_window.clear()
            last_count_time = time.time()
            
        # 3. Display frame & handle exit key
        cv2.imshow("Enhanced ANPR Engine - Live View", frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_feed(source=0)