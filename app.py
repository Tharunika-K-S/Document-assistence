from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import os
import re
import cv2
import numpy as np
import pytesseract

from PIL import Image


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

OCR_LANGUAGE = "tam+eng"


# ============================================================
# DOCUMENT TYPES
# ============================================================

DOCUMENT_TYPES = {

    "Birth Certificate":
        "பிறப்புச் சான்றிதழ்",

    "Death Certificate":
        "இறப்புச் சான்றிதழ்",

    "Income Certificate":
        "வருமானச் சான்றிதழ்",

    "Community Certificate":
        "சமூகச் சான்றிதழ்",

    "Legal Heir Certificate":
        "சட்ட வாரிசுச் சான்றிதழ்",

    "Land / Property Document":
        "நிலம் / சொத்து ஆவணம்",

    "Government Notice / Order":
        "அரசு அறிவிப்பு / உத்தரவு",

    "Court Order":
        "நீதிமன்ற உத்தரவு",

    "Application Form":
        "விண்ணப்பப் படிவம்",

    "General Government Document":
        "பொது அரசு ஆவணம்"
}


# ============================================================
# DOCUMENT CLASSIFICATION KEYWORDS
# ============================================================

CLASSIFICATION_KEYWORDS = {

    "Birth Certificate": [

        "birth certificate",
        "date of birth",
        "place of birth",
        "born",
        "registration of birth",
        "பிறப்புச் சான்றிதழ்",
        "பிறந்த தேதி",
        "பிறந்த இடம்",
        "பிறப்பு"
    ],


    "Death Certificate": [

        "death certificate",
        "date of death",
        "place of death",
        "deceased",
        "cause of death",
        "registration of death",
        "இறப்புச் சான்றிதழ்",
        "இறந்த தேதி",
        "இறந்த இடம்",
        "இறப்பு"
    ],


    "Income Certificate": [

        "income certificate",
        "annual income",
        "income",
        "salary",
        "family income",
        "வருமானச் சான்றிதழ்",
        "ஆண்டு வருமானம்",
        "வருமானம்"
    ],


    "Community Certificate": [

        "community certificate",
        "community",
        "caste",
        "scheduled caste",
        "scheduled tribe",
        "backward class",
        "மூலமாக",
        "சமூகச் சான்றிதழ்",
        "சமூகம்",
        "சாதி"
    ],


    "Legal Heir Certificate": [

        "legal heir",
        "legal heirs",
        "heir certificate",
        "surviving member",
        "deceased person",
        "வாரிசு",
        "சட்ட வாரிசு",
        "சட்ட வாரிசுச் சான்றிதழ்",
        "இறந்தவர்"
    ],


    "Land / Property Document": [

        "survey number",
        "patta",
        "property",
        "land",
        "extent",
        "village",
        "taluk",
        "sub division",
        "document number",
        "சர்வே எண்",
        "பட்டா",
        "நிலம்",
        "சொத்து",
        "கிராமம்",
        "வட்டம்"
    ],


    "Government Notice / Order": [

        "government order",
        "government notice",
        "department",
        "proceedings",
        "notification",
        "official order",
        "அரசாணை",
        "அரசு உத்தரவு",
        "அறிவிப்பு",
        "துறை",
        "சுற்றறிக்கை"
    ],


    "Court Order": [

        "court order",
        "court",
        "high court",
        "district court",
        "supreme court",
        "judgment",
        "judgement",
        "order",
        "petitioner",
        "respondent",
        "plaintiff",
        "defendant",
        "appellant",
        "case no",
        "case number",
        "disposed",
        "allowed",
        "dismissed",
        "interim order",
        "final order",
        "direction",
        "directions",
        "justice",
        "judge",
        "magistrate",
        "sessions court",
        "civil court",
        "family court",
        "நீதிமன்றம்",
        "உயர்நீதிமன்றம்",
        "உச்சநீதிமன்றம்",
        "நீதிபதி",
        "மனுதாரர்",
        "எதிர்மனுதாரர்",
        "வழக்கு எண்",
        "தீர்ப்பு",
        "உத்தரவு",
        "முடிவு",
        "அனுமதிக்கப்படுகிறது",
        "தள்ளுபடி",
        "வழக்கு"
    ],


    "Application Form": [

        "application form",
        "application",
        "applicant",
        "signature",
        "mobile number",
        "applicant name",
        "விண்ணப்பப் படிவம்",
        "விண்ணப்பம்",
        "விண்ணப்பதாரர்",
        "கையொப்பம்"
    ]
}


# ============================================================
# ALLOWED FILE
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# IMAGE READ
# ============================================================

def read_image(path):

    image = cv2.imread(path)

    if image is None:
        raise ValueError("Unable to read uploaded image.")

    return image


# ============================================================
# DOCUMENT PERSPECTIVE DETECTION
# ============================================================

def detect_document(image):

    original = image.copy()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
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

    height, width = image.shape[:2]

    image_area = height * width

    for contour in contours[:20]:

        area = cv2.contourArea(contour)

        if area < image_area * 0.20:
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

            points = approx.reshape(4, 2)

            warped = four_point_transform(
                original,
                points
            )

            return warped, True

    return original, False


# ============================================================
# FOUR POINT TRANSFORM
# ============================================================

def order_points(points):

    rect = np.zeros(
        (4, 2),
        dtype="float32"
    )

    s = points.sum(axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(
        points,
        axis=1
    )

    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def four_point_transform(image, points):

    rect = order_points(
        points.astype("float32")
    )

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(
        br - bl
    )

    width_b = np.linalg.norm(
        tr - tl
    )

    max_width = max(
        int(width_a),
        int(width_b)
    )

    height_a = np.linalg.norm(
        tr - br
    )

    height_b = np.linalg.norm(
        tl - bl
    )

    max_height = max(
        int(height_a),
        int(height_b)
    )

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height
        )
    )

    return warped


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def resize_for_ocr(image):

    height, width = image.shape[:2]

    max_dimension = max(
        height,
        width
    )

    if max_dimension < 1800:

        scale = 1800 / max_dimension

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    return image


def create_processing_variants(image):

    image = resize_for_ocr(image)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )[1]

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return {

        "Grayscale":
            gray,

        "CLAHE":
            enhanced,

        "OTSU":
            otsu,

        "Adaptive":
            adaptive
    }


# ============================================================
# OCR
# ============================================================

def perform_ocr(image, psm):

    config = f"--oem 3 --psm {psm}"

    try:

        data = pytesseract.image_to_data(
            image,
            lang=OCR_LANGUAGE,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        texts = []

        confidences = []

        for i in range(
            len(data["text"])
        ):

            text = data["text"][i].strip()

            confidence = data["conf"][i]

            try:
                confidence = float(
                    confidence
                )
            except:
                confidence = -1

            if text:

                texts.append(text)

                if confidence >= 0:
                    confidences.append(
                        confidence
                    )

        text = " ".join(texts)

        confidence = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0
        )

        return text, confidence

    except Exception:

        text = pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGE,
            config=config
        )

        return text, 0


# ============================================================
# LINE BASED OCR
# ============================================================

def perform_line_ocr(image, psm=6):

    config = f"--oem 3 --psm {psm}"

    text = pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGE,
        config=config
    )

    return text


# ============================================================
# OCR CLEANING
# ============================================================

def clean_ocr_text(text):

    if not text:
        return ""

    text = text.replace(
        "\x0c",
        " "
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"[ \t]+",
            " ",
            line
        )

        lines.append(line)

    return "\n".join(lines)


# ============================================================
# OCR QUALITY
# ============================================================

def get_quality(confidence):

    if confidence >= 85:
        return "Excellent"

    if confidence >= 70:
        return "Good"

    if confidence >= 50:
        return "Moderate"

    return "Low"


# ============================================================
# DOCUMENT CLASSIFICATION
# ============================================================

def classify_document(text):

    if not text:
        return ("General Government Document", 0, [])

    lower_text = text.lower()

    exact_phrases = [
        ("Court Order", ["court order", "judgment", "judgement", "நீதிமன்ற உத்தரவு"]),
        ("Birth Certificate", ["birth certificate", "பிறப்புச் சான்றிதழ்"]),
        ("Death Certificate", ["death certificate", "இறப்புச் சான்றிதழ்"]),
        ("Income Certificate", ["income certificate", "income certificate no", "வருமானச் சான்றிதழ்"]),
        ("Community Certificate", ["community certificate", "caste certificate", "சமூகச் சான்றிதழ்"]),
        ("Legal Heir Certificate", ["legal heir certificate", "heir certificate", "சட்ட வாரிசுச் சான்றிதழ்"]),
        ("Land / Property Document", ["sale deed", "property registration", "registration deed", "land document", "property document"]),
        ("Government Notice / Order", ["government order", "government notice", "official order", "government notification", "அரசாணை", "அரசு உத்தரவு", "அரசு அறிவிப்பு"]),
        ("Application Form", ["application form", "விண்ணப்பப் படிவம்"])
    ]

    for document_type, phrases in exact_phrases:
        for phrase in phrases:
            if phrase.lower() in lower_text:
                return (document_type, 99.0, [phrase])

    indicators = {
        "Birth Certificate": ["date of birth", "place of birth", "registration of birth", "born", "பிறந்த தேதி", "பிறந்த இடம்", "பிறப்பு"],
        "Death Certificate": ["date of death", "place of death", "deceased", "cause of death", "registration of death", "இறந்த தேதி", "இறந்த இடம்", "இறப்பு"],
        "Income Certificate": ["annual income", "annual family income", "family income", "income", "salary", "yearly income", "ஆண்டு வருமானம்", "குடும்ப வருமானம்", "வருமானம்"],
        "Community Certificate": ["community", "caste", "scheduled caste", "scheduled tribe", "backward class", "சமூகம்", "சாதி"],
        "Legal Heir Certificate": ["legal heir", "legal heirs", "surviving member", "deceased person", "வாரிசு", "சட்ட வாரிசு", "இறந்தவர்"],
        "Land / Property Document": ["survey number", "survey no", "patta", "chitta", "property", "land", "extent", "sub division", "sale deed", "sub registrar", "சர்வே எண்", "பட்டா", "சிட்டா", "நிலம்", "சொத்து", "சார் பதிவாளர்"],
        "Government Notice / Order": ["government order", "government notice", "official order", "government notification", "proceedings", "notification", "அரசாணை", "அரசு உத்தரவு", "அரசு அறிவிப்பு", "அறிவிப்பு", "சுற்றறிக்கை"],
        "Court Order": ["court", "high court", "district court", "supreme court", "petitioner", "respondent", "case number", "case no", "plaintiff", "defendant", "நீதிமன்றம்", "மனுதாரர்", "எதிர்மனுதாரர்", "வழக்கு எண்"],
        "Application Form": ["application", "applicant", "application number", "signature of applicant", "விண்ணப்பம்", "விண்ணப்பதாரர்", "விண்ணப்ப எண்"]
    }

    matches_by_type = {}
    scores = {}

    for document_type, keywords in indicators.items():
        matches = [k for k in keywords if k.lower() in lower_text]
        matches_by_type[document_type] = matches
        scores[document_type] = len(matches)

    income_matches = matches_by_type["Income Certificate"]
    land_matches = matches_by_type["Land / Property Document"]

    if len(income_matches) >= 2:
        return ("Income Certificate", round(min(75 + len(income_matches) * 4, 95), 2), income_matches)

    if len(land_matches) >= 2:
        return ("Land / Property Document", round(min(70 + len(land_matches) * 5, 95), 2), land_matches)

    court_matches = matches_by_type["Court Order"]
    if len(court_matches) >= 2:
        return ("Court Order", round(min(70 + len(court_matches) * 4, 95), 2), court_matches)

    if not scores or max(scores.values()) == 0:
        return ("General Government Document", 0, [])

    best_type = max(scores, key=scores.get)
    best_matches = matches_by_type[best_type]
    confidence = min((len(best_matches) / max(len(indicators[best_type]), 1)) * 100, 90)

    return (best_type, round(confidence, 2), best_matches)


# ============================================================
# GENERAL FIELD HELPERS
# ============================================================

NOT_DETECTED = "Not detected"


def first_match(patterns, text):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE |
            re.MULTILINE
        )

        if match:

            value = match.group(1).strip()

            value = clean_extracted_value(
                value
            )

            if value:
                return value

    return NOT_DETECTED


def clean_extracted_value(value):

    value = value.strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    value = value.strip(
        " :-|,.;"
    )

    return value


def extract_dates(text):

    patterns = [

        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",

        r"\b\d{1,2}[.]\d{1,2}[.]\d{2,4}\b",

        r"\b\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\b",

        r"\b\d{1,2}\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\.?\s+\d{4}\b"
    ]

    dates = []

    for pattern in patterns:

        found = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        dates.extend(found)

    unique = []

    for date in dates:

        if date not in unique:
            unique.append(date)

    return unique


# ============================================================
# GENERAL FIELD EXTRACTION
# ============================================================

def extract_general_fields(
    text,
    document_type
):

    fields = {}

    if document_type == "Birth Certificate":

        fields["Name"] = first_match(
            [
                r"(?:Name of Child|Child Name|Name)\s*[:\-]\s*(.+)",
                r"(?:பெயர்|குழந்தையின் பெயர்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Father's Name"] = first_match(
            [
                r"(?:Father(?:'s)? Name|Father Name)\s*[:\-]\s*(.+)",
                r"(?:தந்தையின் பெயர்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Mother's Name"] = first_match(
            [
                r"(?:Mother(?:'s)? Name|Mother Name)\s*[:\-]\s*(.+)",
                r"(?:தாயின் பெயர்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Date of Birth"] = first_match(
            [
                r"(?:Date of Birth|DOB)\s*[:\-]\s*(.+)",
                r"(?:பிறந்த தேதி)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Place of Birth"] = first_match(
            [
                r"(?:Place of Birth)\s*[:\-]\s*(.+)",
                r"(?:பிறந்த இடம்)\s*[:\-]\s*(.+)"
            ],
            text
        )


    elif document_type == "Death Certificate":

        fields["Name of Deceased"] = first_match(
            [
                r"(?:Name of Deceased|Deceased Person|Name)\s*[:\-]\s*(.+)",
                r"(?:இறந்தவரின் பெயர்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Date of Death"] = first_match(
            [
                r"(?:Date of Death|DOD)\s*[:\-]\s*(.+)",
                r"(?:இறந்த தேதி)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Place of Death"] = first_match(
            [
                r"(?:Place of Death)\s*[:\-]\s*(.+)",
                r"(?:இறந்த இடம்)\s*[:\-]\s*(.+)"
            ],
            text
        )


    elif document_type == "Income Certificate":

        fields["Name"] = first_match(
            [
                r"(?:Applicant Name|Name of Applicant|Name)\s*[:\-]?\s*(.+)",
                r"(?:விண்ணப்பதாரர் பெயர்|விண்ணப்பதாரரின் பெயர்|பெயர்)\s*[:\-]?\s*(.+)"
            ], text
        )

        fields["Father / Husband Name"] = first_match(
            [
                r"(?:Father / Husband Name|Father's Name|Father Name|Husband Name|Father/Husband Name)\s*[:\-]?\s*(.+)",
                r"(?:தந்தை / கணவர் பெயர்|தந்தையின் பெயர்|தந்தை பெயர்|கணவர் பெயர்)\s*[:\-]?\s*(.+)"
            ], text
        )

        fields["Annual Income"] = first_match(
            [
                r"(?:Annual Family Income|Annual Income|Yearly Income|Family Income|Income)\s*[:\-]?\s*(.+)",
                r"(?:குடும்ப ஆண்டு வருமானம்|ஆண்டு குடும்ப வருமானம்|ஆண்டு வருமானம்|குடும்ப வருமானம்|வருமானம்)\s*[:\-]?\s*(.+)"
            ], text
        )

        fields["Address"] = first_match(
            [
                r"(?:Permanent Address|Residential Address|Address)\s*[:\-]?\s*(.+)",
                r"(?:நிரந்தர முகவரி|வசிப்பிட முகவரி|முகவரி)\s*[:\-]?\s*(.+)"
            ], text
        )

        fields["District"] = first_match(
            [
                r"(?:District Name|District)\s*[:\-]?\s*(.+)",
                r"(?:மாவட்டத்தின் பெயர்|மாவட்டம்)\s*[:\-]?\s*(.+)"
            ], text
        )

        fields["Certificate Number"] = first_match(
            [
                r"(?:Income Certificate No\.?|Certificate No\.?|Certificate Number|Application No\.?|Application Number)\s*[:\-]?\s*(.+)",
                r"(?:வருமானச் சான்றிதழ் எண்|சான்றிதழ் எண்|விண்ணப்ப எண்)\s*[:\-]?\s*(.+)"
            ], text
        )

    elif document_type == "Community Certificate":

        fields["Name"] = first_match(
            [
                r"(?:Name)\s*[:\-]\s*(.+)",
                r"(?:பெயர்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Community"] = first_match(
            [
                r"(?:Community|Caste)\s*[:\-]\s*(.+)",
                r"(?:சமூகம்|சாதி)\s*[:\-]\s*(.+)"
            ],
            text
        )


    elif document_type == "Legal Heir Certificate":

        fields["Deceased Person"] = first_match(
            [
                r"(?:Deceased Person|Name of Deceased)\s*[:\-]\s*(.+)",
                r"(?:இறந்தவர்|இறந்தவரின் பெயர்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Legal Heirs"] = first_match(
            [
                r"(?:Legal Heirs|Legal Heir)\s*[:\-]\s*(.+)",
                r"(?:சட்ட வாரிசுகள்|வாரிசுகள்)\s*[:\-]\s*(.+)"
            ],
            text
        )


    elif document_type == "Land / Property Document":

        fields["Owner Name"] = first_match(
            [
                r"(?:Owner Name|Name of Owner)\s*[:\-]\s*(.+)",
                r"(?:உரிமையாளர் பெயர்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Survey Number"] = first_match(
            [
                r"(?:Survey Number|Survey No\.?)\s*[:\-]?\s*(.+)",
                r"(?:சர்வே எண்)\s*[:\-]?\s*(.+)"
            ],
            text
        )

        fields["Patta Number"] = first_match(
            [
                r"(?:Patta Number|Patta No\.?)\s*[:\-]?\s*(.+)",
                r"(?:பட்டா எண்)\s*[:\-]?\s*(.+)"
            ],
            text
        )

        fields["Village"] = first_match(
            [
                r"(?:Village)\s*[:\-]\s*(.+)",
                r"(?:கிராமம்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Taluk"] = first_match(
            [
                r"(?:Taluk)\s*[:\-]\s*(.+)",
                r"(?:வட்டம்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["District"] = first_match(
            [
                r"(?:District)\s*[:\-]\s*(.+)",
                r"(?:மாவட்டம்)\s*[:\-]\s*(.+)"
            ],
            text
        )


    elif document_type == "Government Notice / Order":

        fields["Order / Notice Number"] = first_match(
            [
                r"(?:Order No\.?|Order Number|Notice No\.?|Notice Number)\s*[:\-]?\s*(.+)",
                r"(?:உத்தரவு எண்|அறிவிப்பு எண்)\s*[:\-]?\s*(.+)"
            ],
            text
        )

        fields["Department"] = first_match(
            [
                r"(?:Department)\s*[:\-]\s*(.+)",
                r"(?:துறை)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Subject"] = first_match(
            [
                r"(?:Subject)\s*[:\-]\s*(.+)",
                r"(?:பொருள்)\s*[:\-]\s*(.+)"
            ],
            text
        )


    elif document_type == "Application Form":

        fields["Applicant Name"] = first_match(
            [
                r"(?:Applicant Name|Name of Applicant|Name)\s*[:\-]\s*(.+)",
                r"(?:விண்ணப்பதாரர் பெயர்|பெயர்)\s*[:\-]\s*(.+)"
            ],
            text
        )

        fields["Address"] = first_match(
            [
                r"(?:Address)\s*[:\-]\s*(.+)",
                r"(?:முகவரி)\s*[:\-]\s*(.+)"
            ],
            text
        )


    else:

        fields["Name"] = first_match(
            [
                r"(?:Name)\s*[:\-]\s*(.+)",
                r"(?:பெயர்)\s*[:\-]\s*(.+)"
            ],
            text
        )


    # Common fields

    dates = extract_dates(text)

    fields["Dates"] = (
        ", ".join(dates[:5])
        if dates
        else NOT_DETECTED
    )

    return fields


# ============================================================
# COURT ORDER - HELPER
# ============================================================

def court_lines(text):

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:

            line = re.sub(
                r"\s+",
                " ",
                line
            )

            lines.append(line)

    return lines


# ============================================================
# COURT ORDER - CASE NUMBER
# ============================================================

def find_case_number(text):

    patterns = [

        r"\b(?:Case\s*No\.?|Case\s*Number)\s*[:\-]?\s*([A-Za-z0-9./()\-]+(?:\s+of\s+\d{4})?)",

        r"\b(?:W\.?\s*P\.?|W\.?\s*P\.?\s*\([A-Z]+\)|Crl\.?\s*O\.?\s*P\.?|O\.?\s*S\.?|S\.?\s*A\.?|C\.?\s*M\.?\s*A\.?|A\.?\s*S\.?|R\.?\s*C\.?\s*S\.?|C\.?\s*C\.?)\s*(?:No\.?|Number)?\s*([A-Za-z0-9./()\-]+(?:\s+of\s+\d{4})?)",

        r"(?:வழக்கு\s*எண்)\s*[:\-]?\s*([A-Za-z0-9./()\-]+(?:\s+of\s+\d{4})?)"
    ]

    result = first_match(
        patterns,
        text
    )

    if result != NOT_DETECTED:

        result = re.sub(
            r"\s+(?=(Petitioner|Respondent|Plaintiff|Defendant)\b).*",
            "",
            result,
            flags=re.IGNORECASE
        )

        return result

    return NOT_DETECTED


# ============================================================
# COURT ORDER - LABELED PERSON
# ============================================================

def find_labeled_person(
    text,
    labels
):

    lines = court_lines(text)

    for line in lines:

        for label in labels:

            pattern = (
                r"^\s*"
                + re.escape(label)
                + r"\s*[:\-]\s*(.+)$"
            )

            match = re.search(
                pattern,
                line,
                flags=re.IGNORECASE
            )

            if match:

                value = clean_extracted_value(
                    match.group(1)
                )

                if value:
                    return value

    return NOT_DETECTED


# ============================================================
# COURT ORDER - COURT NAME
# ============================================================

def find_court_name(text):

    lines = court_lines(text)

    court_terms = [

        "high court",
        "district court",
        "supreme court",
        "sessions court",
        "civil court",
        "family court",
        "magistrate court",
        "judicial magistrate",
        "metropolitan magistrate",
        "court of",
        "நீதிமன்றம்",
        "உயர்நீதிமன்றம்",
        "மாவட்ட நீதிமன்றம்",
        "குடும்ப நீதிமன்றம்"
    ]

    for line in lines[:40]:

        lower = line.lower()

        if any(
            term.lower() in lower
            for term in court_terms
        ):

            return line[:250]

    return NOT_DETECTED


# ============================================================
# COURT ORDER - JUDGE
# ============================================================

def find_judge_name(text):

    lines = court_lines(text)

    judge_labels = [

        "judge",
        "justice",
        "presiding officer",
        "hon'ble judge",
        "honble judge",
        "honourable judge",
        "நீதிபதி",
        "மாண்புமிகு நீதிபதி"
    ]

    for line in lines[:50]:

        lower = line.lower()

        if any(
            label.lower() in lower
            for label in judge_labels
        ):

            # Avoid returning an extremely long OCR line.
            if len(line) <= 250:
                return line

    return NOT_DETECTED


# ============================================================
# COURT ORDER - DATE
# ============================================================

def find_order_date(text):

    patterns = [

        r"(?:Date\s+of\s+Order|Order\s+Date|Date\s+of\s+Judgment)\s*[:\-]\s*(.+)",

        r"(?:ORDER\s+DATED|DATED)\s*[:\-]?\s*(.+)",

        r"(?:உத்தரவு\s+தேதி|தீர்ப்பு\s+தேதி)\s*[:\-]\s*(.+)"
    ]

    result = first_match(
        patterns,
        text
    )

    if result != NOT_DETECTED:

        # Keep first reasonable portion.
        return result[:100]

    dates = extract_dates(text)

    if dates:
        return dates[0]

    return NOT_DETECTED


# ============================================================
# COURT ORDER - CASE PARTIES
# ============================================================

def find_petitioner(text):

    return find_labeled_person(
        text,
        [
            "Petitioner",
            "Petitioner Name",
            "Petitioners",
            "Plaintiff",
            "Plaintiffs",
            "Appellant",
            "Appellants",
            "மனுதாரர்",
            "மனுதாரரின் பெயர்"
        ]
    )


def find_respondent(text):

    return find_labeled_person(
        text,
        [
            "Respondent",
            "Respondent Name",
            "Respondents",
            "Defendant",
            "Defendants",
            "Respondent(s)",
            "எதிர்மனுதாரர்",
            "எதிர்மனுதாரரின் பெயர்"
        ]
    )


# ============================================================
# COURT ORDER - OUTCOME
# ============================================================

def detect_court_outcome(text):

    lines = court_lines(text)

    outcomes = [

        (
            "Partly Allowed",
            [
                "partly allowed",
                "partially allowed",
                "partly allowed",
                "பகுதியளவில் அனுமதிக்கப்படுகிறது",
                "பகுதி அனுமதி"
            ]
        ),

        (
            "Dismissed",
            [
                "petition is dismissed",
                "petition dismissed",
                "petition stands dismissed",
                "dismissed",
                "தள்ளுபடி செய்யப்படுகிறது",
                "தள்ளுபடி"
            ]
        ),

        (
            "Allowed",
            [
                "petition is allowed",
                "petition allowed",
                "allowed",
                "அனுமதிக்கப்படுகிறது",
                "அனுமதிக்கப்பட்டது"
            ]
        ),

        (
            "Disposed",
            [
                "petition is disposed",
                "petition disposed",
                "disposed of",
                "disposed",
                "வழக்கு முடிக்கப்பட்டது",
                "வழக்கு முடிவு"
            ]
        ),

        (
            "Withdrawn",
            [
                "petition withdrawn",
                "withdrawn",
                "திரும்பப் பெறப்பட்டது"
            ]
        ),

        (
            "Rejected",
            [
                "rejected",
                "rejection",
                "நிராகரிக்கப்படுகிறது",
                "நிராகரிக்கப்பட்டது"
            ]
        ),

        (
            "Granted",
            [
                "application is granted",
                "application granted",
                "granted",
                "அனுமதிக்கப்பட்டது"
            ]
        ),

        (
            "Denied",
            [
                "denied",
                "refused",
                "மறுக்கப்படுகிறது",
                "மறுக்கப்பட்டது"
            ]
        ),

        (
            "Remanded",
            [
                "remanded",
                "remand",
                "மீண்டும் அனுப்பப்படுகிறது"
            ]
        ),

        (
            "Adjourned",
            [
                "adjourned",
                "ஒத்திவைக்கப்படுகிறது"
            ]
        )
    ]

    # Search from bottom because final outcome
    # is often near the end of the order.

    search_lines = (
        lines[-40:]
        if len(lines) > 40
        else lines
    )

    for outcome_name, keywords in outcomes:

        for line in reversed(search_lines):

            lower = line.lower()

            for keyword in keywords:

                if keyword.lower() in lower:

                    return {
                        "name": outcome_name,
                        "source_line": line
                    }

    return {
        "name": NOT_DETECTED,
        "source_line": ""
    }


# ============================================================
# COURT ORDER - IMPORTANT POINTS
# ============================================================

def extract_order_points(text):

    lines = court_lines(text)

    direction_keywords = [

        "directed",
        "hereby directed",
        "ordered",
        "it is ordered",
        "shall",
        "shall be",
        "restrained",
        "granted",
        "dismissed",
        "allowed",
        "disposed",
        "remanded",
        "vacated",
        "stay",
        "injunction",
        "the court directs",

        "உத்தரவிடப்படுகிறது",
        "உத்தரவிட்டது",
        "அனுமதிக்கப்படுகிறது",
        "தள்ளுபடி",
        "ஒத்திவைக்கப்படுகிறது",
        "தடை",
        "அனுமதி"
    ]

    points = []

    # Search mainly in later part of the document.
    candidate_lines = (
        lines[-80:]
        if len(lines) > 80
        else lines
    )

    for line in candidate_lines:

        lower = line.lower()

        if len(line) < 10:
            continue

        if any(
            keyword.lower() in lower
            for keyword in direction_keywords
        ):

            if line not in points:

                points.append(line[:500])

        if len(points) >= 5:
            break

    # If no direction terms were detected,
    # use lines from an ORDER section.

    if not points:

        order_started = False

        for line in lines:

            lower = line.lower()

            if (
                lower in {
                    "order",
                    "judgment",
                    "direction",
                    "disposition",
                    "decision"
                }
                or
                "உத்தரவு" in line
                or
                "தீர்ப்பு" in line
            ):

                order_started = True

                continue

            if order_started:

                if len(line) >= 15:

                    points.append(
                        line[:500]
                    )

                if len(points) >= 5:
                    break

    return points


# ============================================================
# COURT ORDER - SUMMARY
# ============================================================

def simplify_court_order(text):

    case_number = find_case_number(
        text
    )

    court_name = find_court_name(
        text
    )

    judge = find_judge_name(
        text
    )

    order_date = find_order_date(
        text
    )

    petitioner = find_petitioner(
        text
    )

    respondent = find_respondent(
        text
    )

    outcome_info = detect_court_outcome(
        text
    )

    outcome = outcome_info["name"]

    important_points = extract_order_points(
        text
    )


    # --------------------------------------------------------
    # ENGLISH SUMMARY
    # --------------------------------------------------------

    english_summary = []

    english_summary.append(
        "This document was identified as a court order / judgment."
    )


    if case_number != NOT_DETECTED:

        english_summary.append(
            f"Case number detected: {case_number}."
        )

    else:

        english_summary.append(
            "Case number was not clearly detected."
        )


    if order_date != NOT_DETECTED:

        english_summary.append(
            f"Order date detected: {order_date}."
        )

    else:

        english_summary.append(
            "Order date was not clearly detected."
        )


    if petitioner != NOT_DETECTED:

        english_summary.append(
            f"Petitioner / plaintiff information detected: {petitioner}."
        )


    if respondent != NOT_DETECTED:

        english_summary.append(
            f"Respondent / defendant information detected: {respondent}."
        )


    if outcome != NOT_DETECTED:

        english_summary.append(
            f"Detected outcome phrase: {outcome}."
        )

    else:

        english_summary.append(
            "The final outcome was not clearly detected from the OCR text."
        )


    if important_points:

        english_summary.append(
            "Important order-related lines are shown below based on the OCR text."
        )

    else:

        english_summary.append(
            "No clear direction or order point was detected."
        )


    # --------------------------------------------------------
    # TAMIL SUMMARY
    # --------------------------------------------------------

    tamil_summary = []

    tamil_summary.append(
        "இந்த ஆவணம் ஒரு நீதிமன்ற உத்தரவு / தீர்ப்பு ஆவணமாக அடையாளம் காணப்பட்டுள்ளது."
    )


    if case_number != NOT_DETECTED:

        tamil_summary.append(
            f"கண்டறியப்பட்ட வழக்கு எண்: {case_number}."
        )

    else:

        tamil_summary.append(
            "வழக்கு எண் தெளிவாக கண்டறியப்படவில்லை."
        )


    if order_date != NOT_DETECTED:

        tamil_summary.append(
            f"கண்டறியப்பட்ட உத்தரவு தேதி: {order_date}."
        )

    else:

        tamil_summary.append(
            "உத்தரவு தேதி தெளிவாக கண்டறியப்படவில்லை."
        )


    if petitioner != NOT_DETECTED:

        tamil_summary.append(
            f"மனுதாரர் / Plaintiff தகவல்: {petitioner}."
        )


    if respondent != NOT_DETECTED:

        tamil_summary.append(
            f"எதிர்மனுதாரர் / Defendant தகவல்: {respondent}."
        )


    if outcome != NOT_DETECTED:

        tamil_summary.append(
            f"கண்டறியப்பட்ட முடிவு: {outcome}."
        )

    else:

        tamil_summary.append(
            "OCR text அடிப்படையில் இறுதி முடிவு தெளிவாக கண்டறியப்படவில்லை."
        )


    if important_points:

        tamil_summary.append(
            "முக்கியமான உத்தரவு தொடர்பான வரிகள் கீழே OCR text அடிப்படையில் காட்டப்பட்டுள்ளன."
        )

    else:

        tamil_summary.append(
            "தெளிவான உத்தரவு குறிப்புகள் கண்டறியப்படவில்லை."
        )


    return {

        "case_number":
            case_number,

        "court_name":
            court_name,

        "judge":
            judge,

        "order_date":
            order_date,

        "petitioner":
            petitioner,

        "respondent":
            respondent,

        "outcome":
            outcome,

        "important_points":
            important_points,

        "english_summary":
            english_summary,

        "tamil_summary":
            tamil_summary
    }


# ============================================================
# BILINGUAL FIELD LABELS
# ============================================================

FIELD_LABELS_TA = {
    "Name": "பெயர்", "Father's Name": "தந்தையின் பெயர்", "Mother's Name": "தாயின் பெயர்",
    "Father / Husband Name": "தந்தை / கணவர் பெயர்", "Owner Name": "உரிமையாளர் பெயர்",
    "Applicant Name": "விண்ணப்பதாரர் பெயர்", "Deceased Person": "இறந்த நபர்",
    "Name of Deceased": "இறந்தவரின் பெயர்", "Legal Heirs": "சட்ட வாரிசுகள்",
    "Date of Birth": "பிறந்த தேதி", "Date of Death": "இறந்த தேதி",
    "Place of Birth": "பிறந்த இடம்", "Place of Death": "இறந்த இடம்",
    "Annual Income": "ஆண்டு வருமானம்", "Community": "சமூகம்", "Caste": "சாதி",
    "Survey Number": "சர்வே எண்", "Patta Number": "பட்டா எண்", "Village": "கிராமம்",
    "Taluk": "வட்டம்", "District": "மாவட்டம்", "Address": "முகவரி",
    "Certificate Number": "சான்றிதழ் எண்", "Document Number": "ஆவண எண்",
    "Order / Notice Number": "உத்தரவு / அறிவிப்பு எண்", "Department": "துறை",
    "Subject": "பொருள்", "Dates": "தேதிகள்", "Amounts": "தொகைகள்",
    "Case Number": "வழக்கு எண்", "Court Name": "நீதிமன்றம்", "Judge": "நீதிபதி",
    "Order Date": "உத்தரவு தேதி", "Petitioner": "மனுதாரர்", "Respondent": "எதிர்மனுதாரர்",
    "Outcome": "முடிவு", "Important Points": "முக்கியமான அம்சங்கள்"
}


def build_bilingual_fields(fields):
    return [
        {
            "key": key,
            "label_en": key,
            "label_ta": FIELD_LABELS_TA.get(key, key),
            "value": value
        }
        for key, value in fields.items()
    ]


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(
    image_path,
    original_filename
):

    image = read_image(
        image_path
    )


    # --------------------------------------------------------
    # DOCUMENT DETECTION
    # --------------------------------------------------------

    processed_image, document_detected = (
        detect_document(image)
    )


    # --------------------------------------------------------
    # OCR VARIANTS
    # --------------------------------------------------------

    variants = create_processing_variants(
        processed_image
    )


    psm_values = [
        3,
        4,
        6,
        11
    ]


    best_text = ""

    best_confidence = 0

    best_version = "Grayscale"

    best_psm = 6


    # --------------------------------------------------------
    # RUN MULTIPLE OCR COMBINATIONS
    # --------------------------------------------------------

    for version_name, variant in variants.items():

        for psm in psm_values:

            text, confidence = perform_ocr(
                variant,
                psm
            )

            cleaned = clean_ocr_text(
                text
            )

            if len(cleaned.strip()) < 5:
                continue

            # Prefer confidence but also avoid
            # selecting very short OCR output.

            adjusted_score = confidence

            if len(cleaned) < 50:

                adjusted_score *= 0.8

            if adjusted_score > best_confidence:

                best_confidence = confidence

                best_text = cleaned

                best_version = version_name

                best_psm = psm


    # --------------------------------------------------------
    # FALLBACK OCR
    # --------------------------------------------------------

    if not best_text:

        fallback = perform_line_ocr(
            variants["Grayscale"],
            6
        )

        best_text = clean_ocr_text(
            fallback
        )

        best_confidence = 0

        best_version = "Grayscale"

        best_psm = 6


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    (
        document_type,
        classification_confidence,
        classification_keywords
    ) = classify_document(
        best_text
    )


    # --------------------------------------------------------
    # GENERAL FIELDS
    # --------------------------------------------------------

    fields = extract_general_fields(
        best_text,
        document_type
    )


    # --------------------------------------------------------
    # COURT ORDER
    # --------------------------------------------------------

    court_order = None

    if document_type == "Court Order":

        court_order = simplify_court_order(
            best_text
        )

        # Add important court fields into
        # the general extracted-information table.

        fields = {

            "Case Number":
                court_order["case_number"],

            "Court Name":
                court_order["court_name"],

            "Judge":
                court_order["judge"],

            "Order Date":
                court_order["order_date"],

            "Petitioner":
                court_order["petitioner"],

            "Respondent":
                court_order["respondent"],

            "Outcome":
                court_order["outcome"],

            "Important Points":
                (
                    " | ".join(
                        court_order[
                            "important_points"
                        ]
                    )
                    if court_order[
                        "important_points"
                    ]
                    else NOT_DETECTED
                ),

            "Dates":
                (
                    ", ".join(
                        extract_dates(
                            best_text
                        )[:5]
                    )
                    if extract_dates(best_text)
                    else NOT_DETECTED
                )
        }


    # --------------------------------------------------------
    # SAVE PROCESSED IMAGE
    # --------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(
            original_filename
        )
    )[0]

    processed_filename = (
        f"{base_name}_processed.jpg"
    )

    processed_path = os.path.join(
        PROCESSED_FOLDER,
        processed_filename
    )


    cv2.imwrite(
        processed_path,
        processed_image
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "text":
            best_text,

        "confidence":
            round(
                max(
                    0,
                    min(
                        best_confidence,
                        100
                    )
                ),
                2
            ),

        "quality":
            get_quality(
                best_confidence
            ),

        "best_version":
            best_version,

        "best_psm":
            best_psm,

        "document_detected":
            document_detected,

        "processed_filename":
            processed_filename,

        "document_type":
            document_type,

        "document_type_ta":
            DOCUMENT_TYPES.get(
                document_type,
                "பொது அரசு ஆவணம்"
            ),

        "classification_confidence":
            classification_confidence,

        "classification_keywords":
            classification_keywords,

        "fields":
            fields,

        "fields_i18n":
            build_bilingual_fields(fields),

        "court_order":
            court_order
    }


    return result


# ============================================================
# FLASK HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    error = None

    result = None


    if request.method == "POST":

        if "file" not in request.files:

            error = (
                "No file was uploaded."
            )

            return render_template(
                "index.html",
                error=error,
                result=None
            )


        file = request.files["file"]


        if file.filename == "":

            error = (
                "Please select an image."
            )

            return render_template(
                "index.html",
                error=error,
                result=None
            )


        if not allowed_file(
            file.filename
        ):

            error = (
                "Unsupported file type. "
                "Please upload JPG, JPEG or PNG."
            )

            return render_template(
                "index.html",
                error=error,
                result=None
            )


        filename = secure_filename(
            file.filename
        )


        # Avoid problematic duplicate names.

        import time

        unique_filename = (
            f"{int(time.time())}_{filename}"
        )


        upload_path = os.path.join(
            UPLOAD_FOLDER,
            unique_filename
        )


        file.save(
            upload_path
        )


        try:

            result = process_document(
                upload_path,
                unique_filename
            )

        except Exception as exc:

            print(
                "PROCESSING ERROR:",
                exc
            )

            error = (
                "Document processing failed. "
                "Please check the image and "
                "Tesseract installation."
            )


    return render_template(
        "index.html",
        error=error,
        result=result
    )


# ============================================================
# PROCESSED IMAGE ROUTE
# ============================================================

@app.route(
    "/processed/<filename>"
)
def processed_file(filename):

    return send_from_directory(
        PROCESSED_FOLDER,
        filename
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "Tamil + English Government Document "
        "Readability Assistant"
    )

    print("=" * 60)

    print(
        "Tesseract:"
    )

    print(
        TESSERACT_PATH
    )

    print(
        "OCR Language:"
    )

    print(
        OCR_LANGUAGE
    )

    print()

    print(
        "Open browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )