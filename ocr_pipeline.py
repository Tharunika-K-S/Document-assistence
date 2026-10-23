"""
GovEase AI - Robust Bilingual OCR and Image Processing Pipeline
Supports OpenCV preprocessing, deskewing, perspective correction,
multi-variant OCR (tam+eng), and comprehensive PDF (scanned & digital) handling.
"""

import os
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

# Tesseract paths
DEFAULT_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\tharu\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
]

def find_tesseract_cmd():
    for p in DEFAULT_TESSERACT_PATHS:
        if os.path.exists(p):
            return p
    return "tesseract"

TESSERACT_CMD = find_tesseract_cmd()
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def get_installed_languages():
    """Check what languages are installed in Tesseract tessdata."""
    try:
        langs = pytesseract.get_languages(config="")
        return langs
    except Exception:
        # Check tessdata directory directly
        tess_dir = os.path.dirname(TESSERACT_CMD)
        tessdata = os.path.join(tess_dir, "tessdata")
        if os.path.exists(tessdata):
            files = os.listdir(tessdata)
            return [f[:-12] for f in files if f.endswith(".traineddata")]
        return ["eng"]

def get_optimal_ocr_language():
    """Verify if Tamil is present; fallback to English if missing."""
    installed = get_installed_languages()
    if "tam" in installed and "eng" in installed:
        return "tam+eng", True
    elif "tam" in installed:
        return "tam", True
    elif "eng" in installed:
        return "eng", False
    return "eng", False

OCR_LANGUAGE, HAS_TAMIL = get_optimal_ocr_language()

def read_image_safe(path):
    """
    Safely read image from path, handling Windows Unicode / space paths properly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Use numpy fromfile to support arbitrary unicode paths on Windows
    stream = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image from {path}")
    return image

def order_points(points):
    """Order coordinates: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect

def four_point_transform(image, points):
    """Perspective warp document into upright rectangular scan."""
    rect = order_points(points.astype("float32"))
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    return warped

def detect_document_contour(image):
    """Detect document edges and perspective correct if a clear quad contour exists."""
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    height, width = image.shape[:2]
    image_area = height * width

    for contour in contours[:15]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.25:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            points = approx.reshape(4, 2)
            try:
                warped = four_point_transform(original, points)
                # Sanity check: warped image should have reasonable dimensions
                wh, ww = warped.shape[:2]
                if wh > 200 and ww > 200:
                    return warped, True
            except Exception:
                pass

    return original, False

def deskew_image(image):
    """Detect text orientation angle and deskew image."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Invert colors so text is white on black background
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # If angle is minor skew (within +-15 degrees), correct it
        if abs(angle) > 0.5 and abs(angle) < 20.0:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    except Exception:
        pass
    return image

def resize_for_ocr(image):
    """Upscale image if smaller than ideal OCR resolution (at least 1800px on long edge)."""
    height, width = image.shape[:2]
    max_dimension = max(height, width)
    if max_dimension < 1800:
        scale = 1800.0 / max_dimension
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    elif max_dimension > 3600:
        scale = 3600.0 / max_dimension
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image

def create_ocr_variants(image):
    """Generate multiple image enhancement variants for OCR trial."""
    image = resize_for_ocr(image)
    image = deskew_image(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Variant 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_clahe = clahe.apply(gray)

    # Variant 2: Bilateral Filter (reduces noise while keeping edges crisp)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Variant 3: Adaptive Gaussian Thresholding
    adaptive = cv2.adaptiveThreshold(
        enhanced_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    # Variant 4: Otsu Binarization
    otsu = cv2.threshold(enhanced_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Variant 5: Sharpened
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced_clahe, -1, kernel)

    return {
        "CLAHE": enhanced_clahe,
        "Grayscale": gray,
        "Adaptive": adaptive,
        "Bilateral": denoised,
        "Sharpened": sharpened,
        "OTSU": otsu
    }

def clean_ocr_text(text):
    """Normalize extracted OCR text."""
    if not text:
        return ""
    text = text.replace("\x0c", " ").replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"[ \t]+", " ", line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def run_single_ocr(variant_img, psm, lang=None):
    """Execute pytesseract image_to_data and compute confidence."""
    lang = lang or OCR_LANGUAGE
    config = f"--oem 3 --psm {psm}"
    try:
        data = pytesseract.image_to_data(
            variant_img,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT
        )
        texts = []
        confidences = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word = (data["text"][i] or "").strip()
            conf_val = data["conf"][i]
            try:
                conf = float(conf_val)
            except Exception:
                conf = -1.0

            if word:
                texts.append(word)
                if conf >= 0:
                    confidences.append(conf)

        full_text = " ".join(texts)
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
        return full_text, avg_conf
    except Exception as e:
        return "", 0.0

def get_quality_rating(confidence, text_len):
    """Determine user-facing quality rating and warning."""
    if text_len < 20:
        return "Poor", "Low text extraction. The document may be blurry, faded, or handwritten. Please upload a higher resolution scan."
    if confidence >= 75:
        return "Good", None
    elif confidence >= 50:
        return "Moderate", "Some characters or low-contrast text may not have been read with 100% confidence. Please verify critical numbers against the original document."
    else:
        return "Low", "Text quality is low. Some names or numbers may be partially unreadable. Consider re-scanning with higher contrast."

def process_image_file(image_path, original_filename, processed_folder):
    """Full image OCR pipeline execution with multi-variant strategy."""
    image = read_image_safe(image_path)
    processed_image, document_detected = detect_document_contour(image)
    variants = create_ocr_variants(processed_image)

    # PSM candidates: 3 (Auto), 4 (Single column variable sizes), 6 (Uniform block of text), 11 (Sparse text)
    psm_candidates = [6, 4, 3, 11]
    
    best_text = ""
    best_conf = 0.0
    best_version = "CLAHE"
    best_psm = 6

    # Test top variants first for fast turnaround
    priority_variants = ["CLAHE", "Adaptive", "Grayscale", "Sharpened"]
    for ver_name in priority_variants:
        if ver_name not in variants:
            continue
        var_img = variants[ver_name]
        for psm in psm_candidates:
            raw_text, conf = run_single_ocr(var_img, psm)
            cleaned = clean_ocr_text(raw_text)
            if len(cleaned.strip()) < 10:
                continue

            # Score based on confidence and meaningful text volume
            score = conf
            if len(cleaned) < 50:
                score *= 0.75

            if score > best_conf or (not best_text and cleaned):
                best_conf = conf
                best_text = cleaned
                best_version = ver_name
                best_psm = psm

        # If we got a high confidence match (> 80%), break early to keep response fast
        if best_conf >= 80 and len(best_text) >= 100:
            break

    # Fallback if no variant reached threshold
    if not best_text:
        raw_text, conf = run_single_ocr(variants["Grayscale"], 6)
        best_text = clean_ocr_text(raw_text)
        best_conf = conf
        best_version = "Grayscale"
        best_psm = 6

    # Save processed image preview
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    safe_base = re.sub(r"[^A-Za-z0-9_.-]", "_", base_name)
    processed_filename = f"{safe_base}_processed.jpg"
    processed_path = os.path.join(processed_folder, processed_filename)

    cv2.imwrite(processed_path, processed_image)

    quality, warning = get_quality_rating(best_conf, len(best_text))

    return {
        "text": best_text,
        "confidence": round(max(0.0, min(best_conf, 100.0)), 2),
        "quality": quality,
        "warning": warning,
        "best_version": best_version,
        "best_psm": best_psm,
        "document_detected": document_detected,
        "processed_filename": processed_filename,
        "has_tamil": HAS_TAMIL
    }

def process_pdf_file(pdf_path, original_filename, processed_folder):
    """
    Handle PDFs: supports both text-based PDFs and scanned multi-page PDFs
    by converting pages to image pixmaps when embedded text is absent.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed.")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    extracted_text_pages = []
    first_page_image = None
    all_confidences = []
    is_scanned = False

    for page_idx in range(min(total_pages, 8)):
        page = doc[page_idx]
        direct_text = page.get_text().strip()

        # Render page 1 as preview image
        if page_idx == 0:
            pix = page.get_pixmap(dpi=150)
            # Convert pixmap to numpy array
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                first_page_image = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                first_page_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            else:
                first_page_image = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)

        # If direct text exists and has substantial content
        if len(direct_text) > 40:
            extracted_text_pages.append(direct_text)
            all_confidences.append(95.0)
        else:
            # Scanned page - render to high-res image and OCR
            is_scanned = True
            pix_high = page.get_pixmap(dpi=220)
            img_data = np.frombuffer(pix_high.samples, dtype=np.uint8).reshape((pix_high.height, pix_high.width, pix_high.n))
            if pix_high.n >= 3:
                cv_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR if pix_high.n == 3 else cv2.COLOR_RGBA2BGR)
            else:
                cv_img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)

            variants = create_ocr_variants(cv_img)
            text_p, conf_p = run_single_ocr(variants["CLAHE"], 6)
            if not text_p or len(text_p.strip()) < 10:
                text_p, conf_p = run_single_ocr(variants["Adaptive"], 4)
            
            cleaned_p = clean_ocr_text(text_p)
            if cleaned_p:
                extracted_text_pages.append(cleaned_p)
                all_confidences.append(conf_p)

    doc.close()

    full_pdf_text = "\n\n--- Page Break ---\n\n".join(extracted_text_pages)
    overall_conf = (sum(all_confidences) / len(all_confidences)) if all_confidences else 0.0

    # Save page 1 preview image
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    safe_base = re.sub(r"[^A-Za-z0-9_.-]", "_", base_name)
    processed_filename = f"{safe_base}_pdf_preview.jpg"
    processed_path = os.path.join(processed_folder, processed_filename)

    if first_page_image is not None:
        cv2.imwrite(processed_path, first_page_image)
    else:
        processed_filename = None

    quality, warning = get_quality_rating(overall_conf, len(full_pdf_text))

    return {
        "text": full_pdf_text,
        "confidence": round(max(0.0, min(overall_conf, 100.0)), 2),
        "quality": quality,
        "warning": warning,
        "best_version": "Scanned PDF OCR" if is_scanned else "Digital PDF Extract",
        "best_psm": 6 if is_scanned else None,
        "document_detected": True,
        "processed_filename": processed_filename,
        "total_pages": total_pages,
        "has_tamil": HAS_TAMIL
    }
