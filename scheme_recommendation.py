import csv
import os
import re
from flask import Blueprint, jsonify, render_template, request

scheme_bp = Blueprint("scheme_recommendation", __name__, url_prefix="/schemes")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCHEME_FILE_CANDIDATES = [
    "schemes.csv.xls",
    "schemes.csv",
    "schemes.csv(1).xls",
    "schemes.csv(1).csv",
]

_SCHEMES_CACHE = None


def find_scheme_file():
    for filename in SCHEME_FILE_CANDIDATES:
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            return path

    for filename in os.listdir(BASE_DIR):
        low = filename.lower()
        if low.startswith("schemes") and low.endswith((".csv", ".xls", ".xlsx")):
            return os.path.join(BASE_DIR, filename)

    return None


def load_schemes():
    """Load scheme database with in-memory caching to avoid disk latency."""
    global _SCHEMES_CACHE
    if _SCHEMES_CACHE is not None:
        return _SCHEMES_CACHE

    path = find_scheme_file()
    if not path:
        return []

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
            _SCHEMES_CACHE = rows
            return _SCHEMES_CACHE
    except Exception as e:
        print("Error loading schemes file:", e)
        return []


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
        "keywords": ["education", "student", "school", "college", "scholarship", "learning", "graduate", "study"]
    },
    "women_child": {
        "en": "Women and Child / Family Welfare",
        "ta": "பெண்கள், குழந்தைகள் மற்றும் குடும்ப நலம்",
        "keywords": ["women", "woman", "girl", "child", "mother", "maternity", "pregnant", "family"]
    },
    "employment": {
        "en": "Skills & Employment",
        "ta": "திறன் மற்றும் வேலைவாய்ப்பு",
        "keywords": ["employment", "job", "skill", "worker", "training", "livelihood", "career"]
    },
    "business": {
        "en": "Business & Entrepreneurship",
        "ta": "தொழில் மற்றும் தொழில்முனைவு",
        "keywords": ["business", "entrepreneur", "startup", "enterprise", "msme", "self employment", "trade"]
    },
    "finance": {
        "en": "Banking, Financial Services and Insurance",
        "ta": "வங்கி, நிதிச் சேவைகள் மற்றும் காப்பீடு",
        "keywords": ["financial", "finance", "loan", "insurance", "bank", "credit", "pension", "assistance"]
    },
    "housing": {
        "en": "Housing & Shelter",
        "ta": "வீடு மற்றும் குடியிருப்பு",
        "keywords": ["housing", "house", "home", "shelter", "construction", "pradhan mantri awas"]
    },
    "health": {
        "en": "Health & Wellness",
        "ta": "சுகாதாரம் மற்றும் நலன்",
        "keywords": ["health", "medical", "hospital", "disease", "healthcare", "medicine", "ayushman"]
    },
    "senior": {
        "en": "Senior Citizens Welfare",
        "ta": "மூத்த குடிமக்கள் நலம்",
        "keywords": ["senior", "elderly", "old age", "pension", "social welfare", "aged"]
    },
    "disability": {
        "en": "Differently Abled / Disability Support",
        "ta": "மாற்றுத்திறனாளிகள் நலம் மற்றும் ஆதரவு",
        "keywords": ["disability", "disabled", "divyang", "differently abled", "assistive"]
    },
    "other": {
        "en": "All Government Schemes",
        "ta": "அனைத்து அரசு நலத்திட்டங்கள்",
        "keywords": ["citizen", "welfare", "scheme", "government", "benefit"]
    },
}

PROFILE_QUESTIONS = [
    {"id": "age", "type": "number", "en": "Age", "ta": "வயது", "placeholder": "Example: 24"},
    {"id": "gender", "type": "select", "en": "Gender", "ta": "பாலினம்",
     "options": [("female", "Female", "பெண்"), ("male", "Male", "ஆண்"), ("other", "Other", "மற்றவை")]},
    {"id": "income", "type": "number", "en": "Annual family income (₹)", "ta": "ஆண்டு குடும்ப வருமானம் (₹)", "placeholder": "Example: 150000"},
    {"id": "community", "type": "select", "en": "Community", "ta": "சமூகம்",
     "options": [("sc", "SC", "SC"), ("st", "ST", "ST"), ("obc", "OBC / BC / MBC", "OBC / BC / MBC"), ("general", "General", "பொது"), ("other", "Other / Prefer not to say", "மற்றவை / தெரிவிக்க விரும்பவில்லை")]},
    {"id": "occupation", "type": "select", "en": "Occupation", "ta": "தொழில்",
     "options": [("student", "Student", "மாணவர்"), ("farmer", "Farmer", "விவசாயி"), ("agri_labour", "Agricultural Labourer", "விவசாயத் தொழிலாளர்"), ("employee", "Employee", "ஊழியர்"), ("jobseeker", "Job Seeker", "வேலை தேடுபவர்"), ("business", "Business / Self-employed", "தொழில் / சுயதொழில்"), ("homemaker", "Homemaker", "இல்லத்தரசி"), ("senior", "Senior Citizen", "மூத்த குடிமகன்"), ("other", "Other", "மற்றவை")]},
    {"id": "district", "type": "text", "en": "District", "ta": "மாவட்டம்", "placeholder": "Example: Coimbatore / Chennai"},
]

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
    ],
    "women_child": [
        {"id": "profile", "en": "Which describes you best?", "ta": "உங்களை எந்த வகையில் குறிப்பிடலாம்?",
         "options": [("woman", "Woman", "பெண்"), ("girl", "Girl / Young woman", "சிறுமி / இளம் பெண்"),
                     ("mother", "Mother", "தாய்"), ("pregnant", "Pregnant / Lactating", "கர்ப்பிணி / பாலூட்டும் தாய்"),
                     ("child", "Child", "குழந்தை"), ("other", "Other", "மற்றவை")]},
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
                     ("assistance", "Financial assistance / Grant", "நிதி உதவி / மானியம்"), ("other", "Other", "மற்றவை")]},
    ],
    "housing": [
        {"id": "housing_need", "en": "What housing support do you need?", "ta": "எந்த வீட்டு உதவி தேவை?",
         "options": [("new", "New house", "புதிய வீடு"), ("repair", "Repair / Improvement", "பழுது / மேம்பாடு"),
                     ("rent", "Rental / Shelter support", "வாடகை / குடியிருப்பு உதவி"), ("other", "Other", "மற்றவை")]},
    ],
    "health": [
        {"id": "health_need", "en": "What kind of health support are you looking for?", "ta": "எந்த வகையான சுகாதார உதவி தேவை?",
         "options": [("treatment", "Treatment / Hospital care", "மருத்துவ சிகிச்சை"), ("insurance", "Health insurance", "மருத்துவ காப்பீடு"),
                     ("maternal", "Maternal / Child health", "தாய் / குழந்தை நலம்"), ("emergency", "Emergency support", "அவசர உதவி"),
                     ("other", "Other", "மற்றவை")]},
    ],
    "senior": [
        {"id": "support", "en": "What support are you looking for?", "ta": "எந்த உதவி தேவை?",
         "options": [("pension", "Pension", "ஓய்வூதியம்"), ("care", "Care / Support", "பராமரிப்பு / ஆதரவு"), ("other", "Other", "மற்றவை")]},
    ],
    "disability": [
        {"id": "support", "en": "What support are you looking for?", "ta": "எந்த உதவி தேவை?",
         "options": [("education", "Education / Scholarship", "கல்வி / உதவித்தொகை"), ("employment", "Employment / Livelihood", "வேலைவாய்ப்பு"),
                     ("financial", "Financial assistance", "நிதி உதவி"), ("equipment", "Assistive equipment", "உதவி உபகரணங்கள்"),
                     ("other", "Other", "மற்றவை")]},
    ],
}


def evaluate_scheme_match(row, category_key, answers):
    """
    Evaluate relevance score and compile specific reasons why it matched.
    """
    category = CATEGORY_MAP.get(category_key, CATEGORY_MAP["other"])
    
    row_text = " ".join([
        row.get("scheme_name", ""),
        row.get("short_title", ""),
        row.get("categories", ""),
        row.get("sub_categories", ""),
        row.get("tags", ""),
        row.get("target_beneficiaries", ""),
        row.get("beneficiary_type", ""),
        row.get("brief_description", ""),
        row.get("detailed_description", ""),
        row.get("benefits", ""),
        row.get("eligibility", ""),
        row.get("documents_required", ""),
    ]).lower()

    raw_score = 0
    reasons = []

    # 1. Category relevance
    cat_name = category["en"].lower()
    if cat_name in row.get("categories", "").lower():
        raw_score += 10
        reasons.append(f"Official Category matches: {category['en']}")
    else:
        for kw in category["keywords"]:
            if kw.lower() in row_text:
                raw_score += 1.5

    # 2. State matching
    user_state = str(answers.get("state", "")).strip().lower()
    row_state = str(row.get("state", "")).strip().lower()
    if user_state:
        if row_state and row_state == user_state:
            raw_score += 12
            reasons.append(f"State specific scheme for {answers.get('state')}")
        elif row_state in ("central", "all india", "india", "pan india", ""):
            raw_score += 4
            reasons.append("Central / Pan-India scheme accessible across states")
        else:
            raw_score -= 8

    # 3. Profile factors
    occupation = str(answers.get("occupation", "")).strip().lower()
    if occupation and occupation in row_text:
        raw_score += 3
        reasons.append(f"Targeted for beneficiary profile: {occupation.title()}")

    community = str(answers.get("community", "")).strip().lower()
    if community in ("sc", "st") and any(x in row_text for x in ["sc", "st", "scheduled caste", "scheduled tribe", "dalit", "tribal"]):
        raw_score += 3
        reasons.append(f"Special benefits identified for {community.upper()} community")

    gender = str(answers.get("gender", "")).strip().lower()
    if gender == "female" and any(x in row_text for x in ["women", "woman", "girl", "female", "maternity", "widow"]):
        raw_score += 3
        reasons.append("Dedicated provisions for women beneficiaries")

    # 4. Answers semantic match
    for k, v in answers.items():
        if not v or k in ("state", "district", "age", "income"):
            continue
        v_str = str(v).lower()
        if len(v_str) > 2 and v_str in row_text:
            raw_score += 2

    # Calibrate to realistic 0-100 percentage
    if raw_score >= 18:
        pct = min(98, 85 + int(raw_score * 0.5))
    elif raw_score >= 10:
        pct = min(84, 72 + int(raw_score * 0.8))
    elif raw_score >= 5:
        pct = min(71, 60 + int(raw_score * 1.5))
    elif raw_score > 0:
        pct = max(45, min(59, 45 + int(raw_score * 2.0)))
    else:
        pct = 0

    if not reasons and pct > 0:
        reasons.append(f"General match with {category['en']}")

    return pct, reasons


def public_scheme(row, score, reasons):
    return {
        "name": row.get("scheme_name", "Unnamed scheme"),
        "short_title": row.get("short_title", ""),
        "state": row.get("state", "Central / All India"),
        "level": row.get("level", ""),
        "department": row.get("department", "Government Department"),
        "category": row.get("categories", ""),
        "benefit_type": row.get("benefit_type", ""),
        "description": row.get("brief_description", ""),
        "benefits": split_values(row.get("benefits", ""))[:5],
        "eligibility": split_values(row.get("eligibility", ""))[:6],
        "documents": split_values(row.get("documents_required", ""))[:8],
        "application_mode": row.get("application_mode", ""),
        "source_url": row.get("source_url", ""),
        "match_score": score,
        "reasons": reasons[:3],
        "status_label": "Possible Match — Verify Official Eligibility"
    }


@scheme_bp.route("/", methods=["GET"])
def scheme_home():
    return render_template("schemes.html", categories=CATEGORY_MAP, profile_questions=PROFILE_QUESTIONS)


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

    rows = load_schemes()
    if not rows:
        return jsonify({
            "error": "Scheme database was not found. Please ensure schemes.csv.xls is present in the project directory."
        }), 500

    scored = []
    for row in rows:
        pct_score, reasons = evaluate_scheme_match(row, category, answers)
        if pct_score >= 45:
            scored.append((pct_score, reasons, row))

    scored.sort(key=lambda x: (-x[0], x[2].get("scheme_name", "")))

    results = [public_scheme(row, score, reasons) for score, reasons, row in scored[:25]]

    return jsonify({
        "count": len(results),
        "results": results,
        "disclaimer": "Possible Match — Verify Official Eligibility",
        "message": "These schemes may be relevant based on the answers provided.",
        "note": "This is preliminary matching only. Eligibility is not confirmed. Check the official scheme source for current rules and documents."
    })
