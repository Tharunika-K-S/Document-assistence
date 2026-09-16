from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import pytesseract
import os
import uuid


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


# =========================================================
# TESSERACT
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =========================================================
# READ IMAGE
# =========================================================

def read_image(path):

    image = cv2.imread(path)

    if image is None:
        raise ValueError("Could not read image.")

    return image


# =========================================================
# RESIZE
# =========================================================

def resize_for_processing(image):

    height, width = image.shape[:2]

    # Don't make small images smaller
    if width < 1200:

        scale = 1200 / width

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )

    elif width > 1800:

        scale = 1800 / width

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    return image


# =========================================================
# DOCUMENT DETECTION
# =========================================================

def detect_document(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Edge detection
    edges = cv2.Canny(
        blur,
        30,
        120
    )

    # Close broken edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = image.shape[0] * image.shape[1]

    best = None

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.25:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:

            best = approx.reshape(4, 2)

            break

    return best


# =========================================================
# ORDER CORNERS
# =========================================================

def order_points(points):

    points = np.array(
        points,
        dtype=np.float32
    )

    result = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    total = points.sum(axis=1)

    result[0] = points[np.argmin(total)]
    result[2] = points[np.argmax(total)]

    difference = np.diff(
        points,
        axis=1
    )

    result[1] = points[np.argmin(difference)]
    result[3] = points[np.argmax(difference)]

    return result


# =========================================================
# PERSPECTIVE CORRECTION
# =========================================================

def correct_perspective(image, corners):

    corners = order_points(corners)

    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]

    width1 = np.linalg.norm(
        bottom_right - bottom_left
    )

    width2 = np.linalg.norm(
        top_right - top_left
    )

    width = int(
        max(width1, width2)
    )

    height1 = np.linalg.norm(
        top_right - bottom_right
    )

    height2 = np.linalg.norm(
        top_left - bottom_left
    )

    height = int(
        max(height1, height2)
    )

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return corrected


# =========================================================
# FALLBACK CROP
# =========================================================

def fallback_crop(image):

    """
    Used when OpenCV cannot find a four-corner document.

    Removes a small amount of obvious screenshot
    surrounding content instead of destroying the image.
    """

    height, width = image.shape[:2]

    # For screenshots, remove top/bottom UI areas
    top = int(height * 0.18)
    bottom = int(height * 0.94)

    left = int(width * 0.02)
    right = int(width * 0.98)

    cropped = image[
        top:bottom,
        left:right
    ]

    return cropped


# =========================================================
# UPSCALE
# =========================================================

def upscale(image):

    height, width = image.shape[:2]

    new_width = width * 2
    new_height = height * 2

    enlarged = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )

    return enlarged


# =========================================================
# DESKEW
# =========================================================

def deskew(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Binary image only for finding angle
    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )[1]

    coordinates = np.column_stack(
        np.where(thresh > 0)
    )

    if len(coordinates) < 100:

        return image

    angle = cv2.minAreaRect(
        coordinates
    )[-1]

    if angle < -45:

        angle = -(90 + angle)

    else:

        angle = -angle

    # Ignore tiny angles
    if abs(angle) < 0.5:

        return image

    height, width = image.shape[:2]

    center = (
        width // 2,
        height // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# =========================================================
# CREATE OCR IMAGES
# =========================================================

def create_ocr_versions(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------
    # Version 1 - Original grayscale
    # -----------------------------------------

    version1 = gray

    # -----------------------------------------
    # Version 2 - CLAHE
    # -----------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    version2 = clahe.apply(gray)

    # -----------------------------------------
    # Version 3 - OTSU
    # -----------------------------------------

    version3 = cv2.threshold(
        version2,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )[1]

    # -----------------------------------------
    # Version 4 - Adaptive threshold
    # -----------------------------------------

    version4 = cv2.adaptiveThreshold(
        version2,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        version1,
        version2,
        version3,
        version4
    ]


# =========================================================
# OCR WITH CONFIDENCE
# =========================================================

def run_ocr(image):

    # --------------------------------
    # 1. Convert to grayscale safely
    # --------------------------------

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --------------------------------
    # 2. Upscale
    # --------------------------------

    enlarged = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------
    # 3. Noise reduction
    # --------------------------------

    denoised = cv2.fastNlMeansDenoising(
        enlarged,
        None,
        10,
        7,
        21
    )

    # --------------------------------
    # 4. CLAHE enhancement
    # --------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(denoised)

    # --------------------------------
    # 5. OTSU
    # --------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------
    # 6. Adaptive threshold
    # --------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    versions = [
        ("Grayscale", enlarged),
        ("CLAHE", enhanced),
        ("OTSU", otsu),
        ("Adaptive", adaptive)
    ]

    # --------------------------------
    # 7. Different Tesseract layouts
    # --------------------------------

    psm_modes = [3, 4, 6, 11]

    best_text = ""
    best_confidence = 0
    best_version = ""
    best_psm = 0

    # --------------------------------
    # 8. Test every combination
    # --------------------------------

    for version_name, img in versions:

        for psm in psm_modes:

            config = f"--oem 3 --psm {psm}"

            data = pytesseract.image_to_data(
                img,
                lang="tam+eng",
                config=config,
                output_type=pytesseract.Output.DICT
            )

            lines = {}
            confidences = []

            for i in range(len(data["text"])):

                text = data["text"][i].strip()

                try:
                    conf = float(data["conf"][i])
                except:
                    conf = -1

                if text and conf >= 0:

                    # Preserve line structure
                    block = data["block_num"][i]
                    par = data["par_num"][i]
                    line = data["line_num"][i]

                    key = (block, par, line)

                    if key not in lines:
                        lines[key] = []

                    lines[key].append(text)

                    confidences.append(conf)

            # --------------------------------
            # Create text with line breaks
            # --------------------------------

            output_lines = []

            for words in lines.values():
                output_lines.append(" ".join(words))

            current_text = "\n".join(output_lines)

            # --------------------------------
            # Calculate confidence
            # --------------------------------

            if confidences:
                current_confidence = (
                    sum(confidences) / len(confidences)
                )
            else:
                current_confidence = 0

            print(
                f"{version_name} | "
                f"PSM {psm} | "
                f"Confidence: "
                f"{current_confidence:.2f}%"
            )

            # --------------------------------
            # Select best result
            # --------------------------------

            if current_confidence > best_confidence:

                best_confidence = current_confidence
                best_text = current_text
                best_version = version_name
                best_psm = psm

    print("--------------------------------")
    print("BEST OCR")
    print("Version:", best_version)
    print("PSM:", best_psm)
    print("Confidence:", best_confidence)
    print("--------------------------------")

    return best_text, best_confidence
# =========================================================
# CLEAN OCR
# =========================================================

def clean_text(text):

    text = text.replace(
        "|",
        " "
    )

    text = " ".join(
        text.split()
    )

    return text.strip()


# =========================================================
# MAIN PROCESSING
# =========================================================

def process_document(
    input_path,
    output_filename
):

    # -----------------------------------------
    # READ
    # -----------------------------------------

    image = read_image(
        input_path
    )

    # -----------------------------------------
    # RESIZE
    # -----------------------------------------

    image = resize_for_processing(
        image
    )

    # -----------------------------------------
    # DOCUMENT DETECTION
    # -----------------------------------------

    corners = detect_document(
        image
    )

    if corners is not None:

        document = correct_perspective(
            image,
            corners
        )

        detected = True

    else:

        document = fallback_crop(
            image
        )

        detected = False

    # -----------------------------------------
    # UPSCALE
    # -----------------------------------------

    document = upscale(
        document
    )

    # -----------------------------------------
    # DESKEW
    # -----------------------------------------

    document = deskew(
        document
    )

    # -----------------------------------------
    # OCR VERSIONS
    # -----------------------------------------

    versions = create_ocr_versions(
        document
    )

    best_text = ""
    best_confidence = -1
    best_image = None

    # -----------------------------------------
    # TRY ALL VERSIONS
    # -----------------------------------------

    for version in versions:

        text, confidence = run_ocr(
            version
        )

        if confidence > best_confidence:

            best_confidence = confidence
            best_text = text
            best_image = version

    # -----------------------------------------
    # CLEAN TEXT
    # -----------------------------------------

    best_text = clean_text(
        best_text
    )

    # -----------------------------------------
    # SAVE IMAGE
    # -----------------------------------------

    output_path = os.path.join(
        PROCESSED_FOLDER,
        output_filename
    )

    cv2.imwrite(
        output_path,
        best_image
    )

    return {
        "text": best_text,
        "confidence": round(
            best_confidence,
            2
        ),
        "document_detected": detected,
        "processed_file": output_filename
    }


# =========================================================
# FLASK HOME
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    result = None
    error = None

    if request.method == "POST":

        file = request.files.get(
            "document"
        )

        if not file or file.filename == "":

            error = "Please upload an image."

            return render_template(
                "index.html",
                error=error
            )

        allowed = {
            ".jpg",
            ".jpeg",
            ".png"
        }

        extension = os.path.splitext(
            file.filename
        )[1].lower()

        if extension not in allowed:

            error = (
                "Only JPG, JPEG and PNG "
                "images are supported."
            )

            return render_template(
                "index.html",
                error=error
            )

        unique_id = str(
            uuid.uuid4()
        )

        input_filename = (
            unique_id + extension
        )

        output_filename = (
            unique_id + "_processed.png"
        )

        input_path = os.path.join(
            UPLOAD_FOLDER,
            input_filename
        )

        file.save(
            input_path
        )

        try:

            result = process_document(
                input_path,
                output_filename
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# =========================================================
# PROCESSED IMAGE
# =========================================================

@app.route(
    "/processed/<filename>"
)
def processed_file(filename):

    return send_from_directory(
        PROCESSED_FOLDER,
        filename
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )