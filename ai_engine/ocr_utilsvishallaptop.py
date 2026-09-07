import cv2
import re
import numpy as np
import easyocr

# ============================================================
# CONFIGURATION & VALIDATION
# ============================================================
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD",
    "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN",
    "TR", "TS", "UK", "UP", "WB", "TG"
}

STANDARD_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{1,4}$")
BH_RE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$")

print("Loading EasyOCR Engine...")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

def clean_text(text):
    if text is None: return ""
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if text.startswith("IND") or text.startswith("IN0"):
        text = text[3:]
    return text

def is_valid_plate(text):
    return bool(STANDARD_RE.fullmatch(text) or BH_RE.fullmatch(text))

def plate_format_score(text):
    text = clean_text(text)
    if not text: return -100
    score = 0
    if len(text) == 10: score += 30
    if len(text) >= 2 and text[:2] in INDIAN_STATE_CODES: score += 40
    if len(text) >= 4 and text[2:4].isdigit(): score += 25
    if len(text) >= 8 and text[-4:].isdigit(): score += 30
    if is_valid_plate(text): score += 100
    score -= abs(len(text) - 10) * 8
    return score

def position_correction(text):
    text = clean_text(text)
    if len(text) < 4: return text
    chars = list(text)
    letter_corrections = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    digit_corrections = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "T": "7", "B": "8"}

    for i in range(min(2, len(chars))):
        if chars[i] in letter_corrections: chars[i] = letter_corrections[chars[i]]
    for i in range(2, min(4, len(chars))):
        if chars[i] in digit_corrections: chars[i] = digit_corrections[chars[i]]
    if len(chars) >= 8:
        for i in range(len(chars) - 4, len(chars)):
            if chars[i] in digit_corrections: chars[i] = digit_corrections[chars[i]]
    return "".join(chars)

def assemble_ocr_results(results):
    if not results: return "", 0.0
    items = []
    for result in results:
        if len(result) != 3: continue
        bbox, text, confidence = result
        text = clean_text(text)
        if not text: continue
        ys = [point[1] for point in bbox]
        items.append({"text": text, "x": min([p[0] for p in bbox]), "y": sum(ys) / len(ys), "confidence": float(confidence)})

    if not items: return "", 0.0
    items.sort(key=lambda item: item["x"])
    final_text = "".join([i["text"] for i in items])
    avg_conf = float(np.mean([i["confidence"] for i in items]))
    return final_text, avg_conf

# ============================================================
# MEMBER 2's PREPROCESSING PIPELINE
# ============================================================
def create_variants(crop):
    variants = []
    
    # STEP 1: Resize for better understanding
    height = crop.shape[0]
    scale = 3.0 if height < 40 else 2.0 
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    variants.append(("resized_original", resized))
    
    # STEP 2: Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # STEP 3: Contrast Enhancement (Applied to Grayscale)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    variants.append(("enhanced_gray", enhanced_gray))
    
    # STEP 4: Thresholding (Applied to the Enhanced Grayscale)
    _, thresholded = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("thresholded", thresholded))
    
    return variants

def perform_ocr(image):
    try:
        results = reader.readtext(image, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1, paragraph=False, text_threshold=0.4)
        return assemble_ocr_results(results)
    except Exception:
        return "", 0.0

# ============================================================
# MAIN READ FUNCTION
# ============================================================
def read_plate(crop):
    if crop is None or crop.size == 0: return "", 0.0
    
    candidates = []
    
    # Run the image through Member 2's variants
    for name, image in create_variants(crop):
        raw_text, confidence = perform_ocr(image)
        if not raw_text: continue
        
        corrected = position_correction(raw_text)
        score = plate_format_score(corrected)
        
        print(f"[OCR] Filter: {name:<18} | Read: {corrected:<12} | Conf: {confidence:.2f}")
        candidates.append({"text": corrected, "confidence": confidence, "score": score})

        if is_valid_plate(corrected) and confidence > 0.60:
            return corrected, confidence

    if not candidates: return "", 0.0

    # STEP 5: Analyze characters and give best possible output
    counts = {}
    for c in candidates:
        if c["text"]: counts[c["text"]] = counts.get(c["text"], 0) + 1

    best = None
    best_score = -999999
    for c in candidates:
        reps = counts.get(c["text"], 1)
        total_score = c["score"] + (reps * 35) + (c["confidence"] * 20)
        if total_score > best_score:
            best_score = total_score
            best = c

    return best["text"], best["confidence"]