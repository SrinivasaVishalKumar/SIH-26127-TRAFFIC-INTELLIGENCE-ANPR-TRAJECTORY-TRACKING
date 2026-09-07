import cv2
import re
import numpy as np
import easyocr

# ============================================================
# CONFIGURATION & PATTERNS
# ============================================================
OCR_SCALE = 2.0

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD",
    "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK",
    "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN",
    "TR", "TS", "UK", "UP", "WB", "TG"
}

STANDARD_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{1,4}$")
BH_RE = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$")

# ============================================================
# LOAD EASYOCR (Optimized for Intel CPU execution)
# ============================================================
print("Loading EasyOCR Engine...")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)

# ============================================================
# CLEAN & SCORE OCR TEXT
# ============================================================
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    
    # Strip IND / IN0 country identifiers if captured
    if text.startswith("IND") or text.startswith("IN0"):
        text = text[3:]
        
    return text

def is_valid_plate(text):
    return bool(STANDARD_RE.fullmatch(text) or BH_RE.fullmatch(text))

def plate_format_score(text):
    text = clean_text(text)
    if not text:
        return -100
    score = 0
    
    if len(text) == 10:
        score += 30
    if len(text) >= 2 and text[:2] in INDIAN_STATE_CODES:
        score += 40
    if len(text) >= 4 and text[2:4].isdigit():
        score += 25
    if len(text) >= 8 and text[-4:].isdigit():
        score += 30
    if is_valid_plate(text):
        score += 100
    
    score -= abs(len(text) - 10) * 8
    return score

def position_correction(text):
    text = clean_text(text)
    if len(text) < 4:
        return text
    chars = list(text)

    letter_corrections = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    digit_corrections = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "T": "7", "B": "8"}

    # Positions 0 and 1: State code letters
    for i in range(min(2, len(chars))):
        if chars[i] in letter_corrections:
            chars[i] = letter_corrections[chars[i]]
            
    # Positions 2 and 3: District/RTO digits
    for i in range(2, min(4, len(chars))):
        if chars[i] in digit_corrections:
            chars[i] = digit_corrections[chars[i]]
            
    # Trailing positions: Registration numbers
    if len(chars) >= 8:
        for i in range(len(chars) - 4, len(chars)):
            if chars[i] in digit_corrections:
                chars[i] = digit_corrections[chars[i]]
            
    return "".join(chars)

def assemble_ocr_results(results):
    if not results:
        return "", 0.0
    items = []
    for result in results:
        if len(result) != 3:
            continue
        bbox, text, confidence = result
        text = clean_text(text)
        if not text:
            continue
        ys = [point[1] for point in bbox]
        items.append({
            "text": text,
            "x": min([p[0] for p in bbox]),
            "y": sum(ys) / len(ys),
            "height": max(ys) - min(ys),
            "confidence": float(confidence)
        })

    if not items:
        return "", 0.0
    items.sort(key=lambda item: item["x"])
    
    final_text = "".join([i["text"] for i in items])
    avg_conf = float(np.mean([i["confidence"] for i in items]))
    return final_text, avg_conf

# ============================================================
# CREATE OCR IMAGE VARIANTS (Fast & High-Accuracy 3-Filter Suite)
# ============================================================
def create_variants(crop):
    variants = []
    
    height = crop.shape[0]
    if height < 40:
        scale = 3.0  
    elif height > 120:
        scale = 1.0  
    else:
        scale = 2.0  
        
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 1. Original
    variants.append(("original", resized))
    
    # 2. Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))
    
    # 3. Otsu Binarization (High Contrast)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", otsu))
    
    return variants

def perform_ocr(image):
    try:
        results = reader.readtext(
            image,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
            text_threshold=0.4
        )
        return assemble_ocr_results(results)
    except Exception:
        return "", 0.0

# ============================================================
# MAIN PIPELINE FUNCTION
# ============================================================
def read_plate(crop):
    if crop is None or crop.size == 0:
        return "", 0.0
    
    candidates = []
    for name, image in create_variants(crop):
        raw_text, confidence = perform_ocr(image)
        if not raw_text:
            continue
        
        corrected = position_correction(raw_text)
        score = plate_format_score(corrected)
        
        print(f"[TESTING] Filter: {name:<10} | Read: {corrected:<12} | Conf: {confidence:.2f}")
        candidates.append({"text": corrected, "confidence": confidence, "score": score})

        # Early exit on valid match to keep live video smooth
        if is_valid_plate(corrected) and confidence > 0.60:
            print(f"[TESTING] Lock acquired on '{name}' filter.")
            return corrected, confidence

    if not candidates:
        return "", 0.0

    counts = {}
    for c in candidates:
        if c["text"]:
            counts[c["text"]] = counts.get(c["text"], 0) + 1

    best = None
    best_score = -999999
    for c in candidates:
        reps = counts.get(c["text"], 1)
        total_score = c["score"] + (reps * 35) + (c["confidence"] * 20)
        if total_score > best_score:
            best_score = total_score
            best = c

    return best["text"], best["confidence"]