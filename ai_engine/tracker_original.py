import cv2
import os
import time
import json
from datetime import datetime
from ultralytics import YOLO
from ocr_utils import read_plate

# ==========================================
# MAIN VIDEO INFERENCE & AGGREGATION LOOP
# ==========================================
def run_live_feed(source=0, camera_id="CAM_GATE_01"):
    print("Loading License Plate YOLO model (OpenVINO Intel Accelerated)...")
    
    # Loads the OpenVINO IR model directory
    yolo_model = YOLO('license_plate_model_openvino_model') 
    
    cap = cv2.VideoCapture(source)
    # Prevent frame queue lag over network streams
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    last_count_time = time.time()
    unique_plates_in_window = set() 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        
        # Inference scheduled every 5th frame for optimal CPU balance
        if frame_count % 5 == 0: 
            results = yolo_model(frame)
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Apply proportional boundary padding for character retention
                        h, w = frame.shape[:2]
                        px, py = int((x2 - x1) * 0.05), int((y2 - y1) * 0.10)
                        x1, y1 = max(0, x1 - px), max(0, y1 - py)
                        x2, y2 = min(w, x2 + px), min(h, y2 + py)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        cropped = frame[y1:y2, x1:x2]
                        plate_text, confidence = read_plate(cropped)

                        if plate_text and plate_text != "UNKNOWN":
                            unique_plates_in_window.add(plate_text)
                            
                            # Annotate detected registration on live canvas
                            cv2.putText(
                                frame, plate_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                            )
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # --- 15-SECOND BATCH FLUSH TO DATABASE ---
        current_timer = time.time()
        if current_timer - last_count_time >= 15.0:
            if len(unique_plates_in_window) > 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                batch_payload = {
                    "timestamp": current_time,
                    "camera_id": camera_id,
                    "vehicle_count": len(unique_plates_in_window),
                    "detected_plates": list(unique_plates_in_window)
                }
                
                os.makedirs("../database", exist_ok=True)
                batch_json_path = os.path.join("..", "database", "server_payloads.json")
                
                # Maintain valid master JSON array structure
                if os.path.exists(batch_json_path):
                    try:
                        with open(batch_json_path, "r") as file:
                            all_batches = json.load(file)
                    except json.JSONDecodeError:
                        all_batches = []
                else:
                    all_batches = []

                all_batches.append(batch_payload)

                with open(batch_json_path, "w") as file:
                    json.dump(all_batches, file, indent=4)
                    
                print(f"[BATCH SENT] {len(unique_plates_in_window)} cars logged at {current_time}.")
            
            unique_plates_in_window.clear()
            last_count_time = time.time()
        
        cv2.imshow("Enhanced ANPR Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Adjust URL to match your active mobile feed address
    run_live_feed(source="http://192.168.0.114:8080/video", camera_id="CAM_GATE_01")