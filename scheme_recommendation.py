
import csv
import os
import re
from flask import Blueprint, jsonify, render_template, request

scheme_bp = Blueprint("scheme_recommendation", __name__, url_prefix="/schemes")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The uploaded file is CSV data even though its extension is .xls.
SCHEME_FILE_CANDIDATES = [
    "schemes.csv.xls",
    "schemes.csv",
    "schemes.csv(1).xls",
    "schemes.csv(1).csv",
]


def find_scheme_file():
    for filename in SCHEME_FILE_CANDIDATES:
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            return path

    # Also accept a file whose name starts with schemes and ends in .xls/.csv.
    for filename in os.listdir(BASE_DIR):
        low = filename.lower()
        if low.startswith("schemes") and low.endswith((".csv", ".xls", ".xlsx")):
            return os.path.join(BASE_DIR, filename)

    return None


def load_schemes():
    path = find_scheme_file()
    if not path:
        return []

    # Current supplied dataset is comma-separated CSV with an .xls extension.
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
        return rows


def split_values(value):
    if not value:
        return []
    return [x.strip() for x in re.split(r"[;|,]", str(value)) if x.strip()]


CATEGORY_MAP = {
    "agriculture": {
        "en": "Agriculture, Rural & Environment",
        "ta": "விவசாயம், கிராமப்புறம் மற்றும் சுற்றுச்சூழல்",
        "keywords": ["agriculture", "farmer", "farming", "crop", "horticulture", "livestock", "dairy", "fisheries", "rural"]
    },
    "education": {
        "en": "Education & Learning",
        "ta": "கல்வி மற்றும் கற்றல்",
        "keywords": ["education", "student", "school", "college", "scholarship", "learning"]
    },
    "women_child": {
        "en": "Women and Child",
        "ta": "பெண்கள் மற்றும் குழந்தைகள்",
        "keywords": ["women", "woman", "girl", "child", "mother", "maternity", "pregnant"]
    },
    "employment": {
        "en": "Skills & Employment",
        "ta": "திறன் மற்றும் வேலைவாய்ப்பு",
        "keywords": ["employment", "job", "skill", "worker", "training", "livelihood"]
    },
    "business": {
        "en": "Business & Entrepreneurship",
        "ta": "தொழில் மற்றும் தொழில்முனைவு",
        "keywords": ["business", "entrepreneur", "startup", "enterprise", "msme", "self employment"]
    },
    "finance": {
        "en": "Banking, Financial Services and Insurance",
        "ta": "வங்கி, நிதிச் சேவைகள் மற்றும் காப்பீடு",
        "keywords": ["financial", "finance", "loan", "insurance", "bank", "credit", "pension"]
    },
    "housing": {
        "en": "Housing & Shelter",
        "ta": "வீடு மற்றும் குடியிருப்பு",
        "keywords": ["housing", "house", "home", "shelter", "construction"]
    },
    "health": {
        "en": "Health & Wellness",
        "ta": "சுகாதாரம் மற்றும் நலன்",
        "keywords": ["health", "medical", "hospital", "disease", "healthcare", "medicine"]
    },
    "senior": {
        "en": "Social welfare & Empowerment",
        "ta": "சமூக நலன் மற்றும் அதிகாரமளித்தல்",
        "keywords": ["senior", "elderly", "old age", "pension", "social welfare"]
    },
    "disability": {
        "en": "Social welfare & Empowerment",
        "ta": "சமூக நலன் மற்றும் அதிகாரமளித்தல்",
        "keywords": ["disability", "disabled", "divyang", "differently abled"]
    },
    "other": {
        "en": "All categories",
        "ta": "அனைத்து பிரிவுகளும்",
        "keywords": []
    },
}

QUESTIONS = {
    "agriculture": [
        {"id": "farmer_type", "en": "What best describes you?", "ta": "உங்களை எந்த வகை விவசாயி / பயனாளி என்று கூறலாம்?",
         "options": [
             ("farmer", "Farmer", "விவசாயி"), ("agri_labour", "Agricultural Labourer", "விவசாயத் தொழிலாளர்"),
             ("tenant", "Tenant Farmer", "குத்தகை விவசாயி"), ("fpo", "FPO / Farmer Group", "விவசாயிகள் குழு / FPO"),
             ("other", "Other", "மற்றவை")]},
        {"id": "land", "en": "Do you own agricultural land?", "ta": "உங்களிடம் விவசாய நிலம் உள்ளதா?",
         "options": [("yes", "Yes", "ஆம்"), ("no", "No", "இல்லை")]},
        {"id": "activity", "en": "What type of activity are you involved in?", "ta": "எந்த வகை விவசாய நடவடிக்கையில் ஈடுபடுகிறீர்கள்?",
         "options": [("crop", "Crop farming", "பயிர் விவசாயம்"), ("horticulture", "Horticulture", "தோட்டக்கலை"),
                     ("dairy", "Dairy", "பால் பண்ணை"), ("livestock", "Livestock", "கால்நடை"),
                     ("fisheries", "Fisheries", "மீன்வளம்"), ("mixed", "Mixed farming", "கலப்பு விவசாயம்"),
                     ("other", "Other", "மற்றவை")]},
    ],
    "education": [
        {"id": "student", "en": "Are you currently a student?", "ta": "நீங்கள் தற்போது மாணவரா?",
         "options": [("yes", "Yes", "ஆம்"), ("no", "No", "இல்லை")]},
        {"id": "course", "en": "What level are you studying?", "ta": "நீங்கள் எந்த நிலையில் படிக்கிறீர்கள்?",
         "options": [("school", "School", "பள்ளி"), ("diploma", "Diploma", "டிப்ளமோ"),
                     ("ug", "Undergraduate", "இளங்கலை"), ("pg", "Postgraduate", "முதுகலை"),
                     ("professional", "Professional / Technical", "தொழில்முறை / தொழில்நுட்பம்"),
                     ("other", "Other", "மற்றவை")]},
        {"id": "institution", "en": "Type of institution?", "ta": "கல்வி நிறுவனத்தின் வகை?",
         "options": [("government", "Government", "அரசு"), ("aided", "Government-aided", "அரசு உதவி பெறும்"),
                     ("private", "Private", "தனியார்"), ("other", "Other", "மற்றவை")]},
        {"id": "income", "en": "Approximate annual family income?", "ta": "குடும்பத்தின் தோராயமான ஆண்டு வருமானம்?",
         "options": [("below_1l", "Below ₹1 lakh", "₹1 லட்சத்திற்குக் கீழ்"),
                     ("1_2_5l", "₹1–2.5 lakh", "₹1–2.5 லட்சம்"),
                     ("2_5_5l", "₹2.5–5 lakh", "₹2.5–5 லட்சம்"),
                     ("above_5l", "Above ₹5 lakh", "₹5 லட்சத்திற்கு மேல்"),
                     ("unknown", "Not sure", "தெரியவில்லை")]},
    ],
    "women_child": [
        {"id": "profile", "en": "Which describes you best?", "ta": "உங்களை எந்த வகையில் குறிப்பிடலாம்?",
         "options": [("woman", "Woman", "பெண்"), ("girl", "Girl / Young woman", "சிறுமி / இளம் பெண்"),
                     ("mother", "Mother", "தாய்"), ("child", "Child", "குழந்தை"), ("other", "Other", "மற்றவை")]},
    ],
    "employment": [
        {"id": "employment_status", "en": "What is your current situation?", "ta": "உங்கள் தற்போதைய நிலை என்ன?",
         "options": [("jobseeker", "Looking for a job", "வேலை தேடுகிறேன்"), ("worker", "Currently employed", "வேலை செய்கிறேன்"),
                     ("unemployed", "Unemployed", "வேலை இல்லாமல் உள்ளேன்"), ("other", "Other", "மற்றவை")]},
        {"id": "skill", "en": "Are you interested in skill training?", "ta": "திறன் பயிற்சியில் ஆர்வமா?",
         "options": [("yes", "Yes", "ஆம்"), ("no", "No", "இல்லை")]},
    ],
    "business": [
        {"id": "business_stage", "en": "What is your business stage?", "ta": "உங்கள் தொழில் எந்த நிலையில் உள்ளது?",
         "options": [("idea", "Planning / Idea stage", "திட்டம் / யோசனை நிலை"), ("existing", "Existing business", "ஏற்கனவே தொழில் உள்ளது"),
                     ("expansion", "Looking to expand", "விரிவாக்கம் செய்ய விரும்புகிறேன்")]},
    ],
    "finance": [
        {"id": "need", "en": "What type of financial support are you looking for?", "ta": "எந்த வகையான நிதி உதவி தேவை?",
         "options": [("loan", "Loan / Credit", "கடன்"), ("insurance", "Insurance", "காப்பீடு"),
                     ("pension", "Pension / Social security", "ஓய்வூதியம் / சமூக பாதுகாப்பு"),
                     ("assistance", "Financial assistance", "நிதி உதவி"), ("other", "Other", "மற்றவை")]},
    ],
    "housing": [
        {"id": "housing_need", "en": "What housing support do you need?", "ta": "எந்த வீட்டு உதவி தேவை?",
         "options": [("new", "New house", "புதிய வீடு"), ("repair", "Repair / Improvement", "பழுது / மேம்பாடு"),
                     ("rent", "Rental / Shelter support", "வாடகை / குடியிருப்பு உதவி"), ("other", "Other", "மற்றவை")]},
    ],
    "health": [
        {"id": "health_need", "en": "What kind of health support are you looking for?", "ta": "எந்த வகையான சுகாதார உதவி தேவை?",
         "options": [("treatment", "Treatment", "சிகிச்சை"), ("insurance", "Health insurance", "மருத்துவ காப்பீடு"),
                     ("maternal", "Maternal / Child health", "தாய் / குழந்தை நலம்"), ("emergency", "Emergency support", "அவசர உதவி"),
                     ("other", "Other", "மற்றவை")]},
    ],
    "senior": [
        {"id": "support", "en": "What support are you looking for?", "ta": "எந்த உதவி தேவை?",
         "options": [("pension", "Pension", "ஓய்வூதியம்"), ("care", "Care / Support", "பராமரிப்பு / ஆதரவு"), ("other", "Other", "மற்றவை")]},
    ],
    "disability": [
        {"id": "support", "en": "What support are you looking for?", "ta": "எந்த உதவி தேவை?",
         "options": [("education", "Education", "கல்வி"), ("employment", "Employment", "வேலைவாய்ப்பு"),
                     ("financial", "Financial assistance", "நிதி உதவி"), ("equipment", "Assistive equipment", "உதவி உபகரணங்கள்"),
                     ("other", "Other", "மற்றவை")]},
    ],
}


def category_score(row, category_key):
    category = CATEGORY_MAP[category_key]
    text = " ".join([
        row.get("categories", ""),
        row.get("sub_categories", ""),
        row.get("tags", ""),
        row.get("target_beneficiaries", ""),
        row.get("brief_description", ""),
        row.get("detailed_description", ""),
        row.get("eligibility", ""),
        row.get("benefit_type", ""),
    ]).lower()

    score = 0
    for kw in category["keywords"]:
        if kw.lower() in text:
            score += 1

    if category["en"].lower() in row.get("categories", "").lower():
        score += 8

    return score


def answer_score(row, category_key, answers):
    text = " ".join([
        row.get("scheme_name", ""),
        row.get("target_beneficiaries", ""),
        row.get("beneficiary_type", ""),
        row.get("categories", ""),
        row.get("sub_categories", ""),
        row.get("tags", ""),
        row.get("brief_description", ""),
        row.get("detailed_description", ""),
        row.get("benefits", ""),
        row.get("eligibility", ""),
        row.get("documents_required", ""),
    ]).lower()

    score = category_score(row, category_key)

    # Category-specific semantic matching.
    mappings = {
        "agriculture": {
            "farmer": ["farmer", "agricultur"],
            "agri_labour": ["agricultural labour", "farm labour", "labour"],
            "tenant": ["tenant farmer", "tenant", "cultivat"],
            "fpo": ["fpo", "farmer producer", "farmer group"],
            "crop": ["crop", "cultivat", "seed", "fertilizer", "irrigation"],
            "horticulture": ["horticulture", "fruit", "vegetable", "garden"],
            "dairy": ["dairy", "milk", "cattle"],
            "livestock": ["livestock", "animal husbandry", "goat", "sheep", "poultry"],
            "fisheries": ["fish", "fisheries", "aquaculture"],
        },
        "education": {
            "school": ["school", "school student"],
            "diploma": ["diploma", "polytechnic"],
            "ug": ["undergraduate", "graduate", "college", "degree"],
            "pg": ["postgraduate", "post graduate", "master"],
            "professional": ["professional", "technical", "engineering", "medical"],
            "government": ["government institution", "government school", "government college"],
            "aided": ["aided"],
            "scholarship": ["scholarship", "student", "education"],
        },
        "women_child": {
            "woman": ["women", "woman"],
            "girl": ["girl", "female"],
            "mother": ["mother", "maternal", "pregnant", "maternity"],
            "child": ["child", "children", "girl", "boy"],
        },
        "employment": {
            "jobseeker": ["employment", "job", "placement", "job seeker"],
            "worker": ["worker", "employee", "employment"],
            "unemployed": ["unemployed", "employment"],
            "yes": ["skill", "training", "vocational"],
        },
        "business": {
            "idea": ["startup", "entrepreneur", "new enterprise", "business"],
            "existing": ["enterprise", "business", "msme"],
            "expansion": ["expansion", "enterprise", "msme", "business"],
        },
        "finance": {
            "loan": ["loan", "credit", "finance"],
            "insurance": ["insurance"],
            "pension": ["pension", "social security"],
            "assistance": ["financial assistance", "grant", "subsidy", "assistance"],
        },
        "housing": {
            "new": ["housing", "house construction", "new house"],
            "repair": ["repair", "renovation", "improvement"],
            "rent": ["rental", "shelter", "housing"],
        },
        "health": {
            "treatment": ["treatment", "medical", "hospital", "health"],
            "insurance": ["health insurance", "insurance"],
            "maternal": ["maternal", "child health", "maternity"],
            "emergency": ["emergency", "ambulance", "medical emergency"],
        },
        "senior": {
            "pension": ["pension", "old age", "senior citizen"],
            "care": ["elderly", "senior", "old age", "care"],
        },
        "disability": {
            "education": ["disability", "student", "education", "scholarship"],
            "employment": ["disability", "employment", "job"],
            "financial": ["disability", "financial", "pension", "assistance"],
            "equipment": ["assistive", "equipment", "device", "disability"],
        },
    }

    for value in answers.values():
        for kw in mappings.get(category_key, {}).get(value, []):
            if kw in text:
                score += 2

    # State is a strong filter when the dataset identifies a state.
    state = answers.get("state", "").strip().lower()
    if state:
        row_state = row.get("state", "").strip().lower()
        if row_state == state:
            score += 12
        elif row_state in ("central", "all india", "india", "pan india", ""):
            score += 3
        else:
            score -= 10

    return score


def public_scheme(row, score):
    return {
        "name": row.get("scheme_name", "Unnamed scheme"),
        "short_title": row.get("short_title", ""),
        "state": row.get("state", ""),
        "level": row.get("level", ""),
        "department": row.get("department", ""),
        "category": row.get("categories", ""),
        "benefit_type": row.get("benefit_type", ""),
        "description": row.get("brief_description", ""),
        "benefits": split_values(row.get("benefits", ""))[:5],
        "eligibility": split_values(row.get("eligibility", ""))[:6],
        "documents": split_values(row.get("documents_required", ""))[:8],
        "application_mode": row.get("application_mode", ""),
        "source_url": row.get("source_url", ""),
        "match_score": score,
    }


@scheme_bp.route("/", methods=["GET"])
def scheme_home():
    return render_template("schemes.html", categories=CATEGORY_MAP)


@scheme_bp.route("/api/questions/<category>", methods=["GET"])
def scheme_questions(category):
    if category not in CATEGORY_MAP:
        return jsonify({"error": "Unknown category"}), 400
    return jsonify({
        "category": category,
        "questions": QUESTIONS.get(category, [])
    })


@scheme_bp.route("/api/match", methods=["POST"])
def scheme_match():
    data = request.get_json(silent=True) or {}
    category = data.get("category", "")
    answers = data.get("answers", {}) or {}

    if category not in CATEGORY_MAP:
        return jsonify({"error": "Please select a valid category."}), 400

    state = str(answers.get("state", "")).strip()
    rows = load_schemes()

    if not rows:
        return jsonify({
            "error": "Scheme database was not found. Keep the supplied schemes.csv.xls file in the Flask project folder."
        }), 500

    scored = []
    for row in rows:
        score = answer_score(row, category, answers)
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda x: (-x[0], x[1].get("scheme_name", "")))

    # Prefer the strongest matches; avoid presenting hundreds of loosely related schemes.
    results = [public_scheme(row, score) for score, row in scored[:20]]

    return jsonify({
        "count": len(results),
        "results": results,
        "message": "These schemes may be relevant based on the answers provided.",
        "note": "This is preliminary matching only. Eligibility is not confirmed. Check the official scheme source for current rules and documents."
    })
