import cv2
import os
import time
import json
from datetime import datetime
from ultralytics import YOLO
from ocr_utils import read_plate

# ==========================================
# MAIN LIVE VIDEO & BATCH LOGGING ENGINE
# ==========================================
def run_live_feed(source=0, camera_id="CAM_GATE_01"):
    print("Loading License Plate YOLO model (Intel OpenVINO Accelerated)...")
    
    # Load OpenVINO IR model directory
    yolo_model = YOLO('license_plate_model_openvino_model') 
    
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    last_count_time = time.time()
    unique_plates_in_window = set() 
    
    # VISUAL MEMORY: Keeps the green box smooth during skipped frames
    last_detections = [] 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        
        # 1. RUN AI EVERY 5TH FRAME
        if frame_count % 5 == 0: 
            # Added verbose=False to mute the console spam!
            results = yolo_model(frame, verbose=False)
            
            # Clear old memory for the new frame
            last_detections.clear()
            
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Pad coordinates for the OCR engine
                        h, w = frame.shape[:2]
                        px, py = int((x2 - x1) * 0.05), int((y2 - y1) * 0.10)
                        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
                        cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
                        if cx2 <= cx1 or cy2 <= cy1:
                            continue

                        cropped = frame[cy1:cy2, cx1:cx2]
                        plate_text, confidence = read_plate(cropped)

                        if plate_text and plate_text != "UNKNOWN":
                            unique_plates_in_window.add(plate_text)
                            
                        # Save the unpadded YOLO box and text to memory
                        last_detections.append({
                            "box": (x1, y1, x2, y2),
                            "text": plate_text if plate_text else ""
                        })
                        
        # 2. DRAW THE GREEN BOXES ON *EVERY* FRAME
        for det in last_detections:
            bx1, by1, bx2, by2 = det["box"]
            text = det["text"]
            
            # Draw the tight green bounding box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            
            # Draw the text if OCR successfully read it
            if text:
                cv2.putText(frame, text, (bx1, by1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # --- 15-SECOND BATCH EXPORT ---
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
    # Point this to your active smartphone stream URL
    phone_url = "http://192.168.0.114:8080/video"
    run_live_feed(source=0, camera_id="CAM_GATE_01")