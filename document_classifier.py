"""
GovEase AI - Universal Government Document Classification & Field Extraction
Covers all major Tamil Nadu & Indian government documents with bilingual field mapping
and automatic fallback for general government documents.
"""

import re

NOT_DETECTED = "Not Detected"

DOCUMENT_TYPES = {
    "Birth Certificate": "பிறப்புச் சான்றிதழ்",
    "Death Certificate": "இறப்புச் சான்றிதழ்",
    "Income Certificate": "வருமானச் சான்றிதழ்",
    "Community Certificate": "சமூகச் சான்றிதழ்",
    "Legal Heir Certificate": "சட்ட வாரிசுச் சான்றிதழ்",
    "First Graduate Certificate": "முதல் பட்டதாரி சான்றிதழ்",
    "Residence / Nativity Certificate": "இருப்பிடச் சான்றிதழ்",
    "Land / Property Document": "நிலம் / சொத்து ஆவணம்",
    "Government Notice / Order": "அரசு அறிவிப்பு / உத்தரவு",
    "Court Order": "நீதிமன்ற உத்தரவு",
    "Ration Card / Family Card": "குடும்ப அட்டை / ரேஷன் கார்டு",
    "Aadhaar Document": "ஆதார் ஆவணம் (பாதுகாக்கப்பட்டது)",
    "Educational Certificate": "கல்விச் சான்றிதழ் / மதிப்பெண் சான்றிதழ்",
    "Pension / Financial Assistance": "ஓய்வூதியம் / உதவித்தொகை ஆவணம்",
    "Application Form": "விண்ணப்பப் படிவம்",
    "General Government Document": "பொது அரசு ஆவணம்"
}

FIELD_LABEL_TAMIL = {
    "Name": "பெயர்",
    "Applicant Name": "விண்ணப்பதாரர் பெயர்",
    "Child Name": "குழந்தையின் பெயர்",
    "Father's Name": "தந்தையின் பெயர்",
    "Mother's Name": "தாயின் பெயர்",
    "Father / Husband Name": "தந்தை / கணவர் பெயர்",
    "Father / Mother Name": "தந்தை / தாய் பெயர்",
    "Date of Birth": "பிறந்த தேதி",
    "Place of Birth": "பிறந்த இடம்",
    "Name of Deceased": "இறந்தவரின் பெயர்",
    "Date of Death": "இறந்த தேதி",
    "Place of Death": "இறந்த இடம்",
    "Annual Income": "ஆண்டு வருமானம்",
    "Address": "முகவரி",
    "Residence Address": "குடியிருப்பு முகவரி",
    "District": "மாவட்டம்",
    "Taluk": "வட்டம்",
    "Village": "கிராமம்",
    "Certificate Number": "சான்றிதழ் எண்",
    "Community": "சமூகம்",
    "Caste": "சாதி",
    "Deceased Person": "இறந்த நபர்",
    "Legal Heirs": "சட்ட வாரிசுகள்",
    "Relationship": "உறவு",
    "Owner Name": "உரிமையாளர் பெயர்",
    "Survey Number": "சர்வே எண்",
    "Patta Number": "பட்டா எண்",
    "Extent": "பரப்பளவு",
    "Order / Notice Number": "உத்தரவு / அறிவிப்பு எண்",
    "Department": "துறை",
    "Subject": "பொருள்",
    "Application Number": "விண்ணப்ப எண்",
    "Educational Qualification": "கல்வித் தகுதி",
    "Institution / College": "கல்வி நிறுவனம் / கல்லூரி",
    "Ration Card Number": "குடும்ப அட்டை எண்",
    "Card Type": "அட்டை வகை",
    "Aadhaar Number": "ஆதார் எண் (பாதுகாக்கப்பட்டது)",
    "Pension Number": "ஓய்வூதிய எண்",
    "Financial Amount": "தொகை / உதவித்தொகை",
    "Issuing Authority": "வழங்கிய அதிகாரி",
    "Document / Reference Number": "ஆவண / குறிப்பு எண்",
    "Status": "நிலை",
    "Case Number": "வழக்கு எண்",
    "Court Name": "நீதிமன்றம்",
    "Judge": "நீதிபதி",
    "Order Date": "உத்தரவு தேதி",
    "Petitioner": "மனுதாரர்",
    "Respondent": "எதிர்மனுதாரர்",
    "Outcome": "முடிவு",
    "Important Points": "முக்கிய அம்சங்கள்",
    "Dates": "தேதிகள்"
}

def clean_extracted_value(value):
    if not value:
        return NOT_DETECTED
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^[:\-\s]+", "", value)
    value = re.sub(r"[:\-\s]+$", "", value)
    return value if value else NOT_DETECTED

def extract_dates(text):
    date_patterns = [
        r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b",
    ]
    matches = []
    for pat in date_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            d = m.group(0).strip()
            if d not in matches:
                matches.append(d)
    return matches

def first_match_multiline(patterns, text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for pattern in patterns:
        for line in lines:
            m = re.search(pattern, line, flags=re.IGNORECASE)
            if m:
                val = clean_extracted_value(m.group(1))
                if val != NOT_DETECTED and len(val) >= 2:
                    return val
    # Try against full text if single-line regex didn't match
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            val = clean_extracted_value(m.group(1))
            if val != NOT_DETECTED and len(val) >= 2:
                return val
    return NOT_DETECTED

def classify_document(text):
    """
    Classify document into one of the supported government categories.
    Falls back to 'General Government Document' if confidence is low.
    """
    if not text or len(text.strip()) < 15:
        return ("General Government Document", 0.0, [])

    lower = re.sub(r"\s+", " ", text.lower()).strip()

    # Exact Title Priorities
    exact_priority = [
        ("First Graduate Certificate", [
            "first graduate certificate", "first graduate", "முதல் பட்டதாரி சான்றிதழ்", "முதல் பட்டதாரி"
        ]),
        ("Income Certificate", [
            "income certificate", "வருமானச் சான்றிதழ்", "வருமான சான்றிதழ்"
        ]),
        ("Birth Certificate", [
            "birth certificate", "பிறப்புச் சான்றிதழ்", "பிறப்பு சான்றிதழ்"
        ]),
        ("Death Certificate", [
            "death certificate", "இறப்புச் சான்றிதழ்", "இறப்பு சான்றிதழ்"
        ]),
        ("Community Certificate", [
            "community certificate", "caste certificate", "சமூகச் சான்றிதழ்", "சாதிச் சான்றிதழ்"
        ]),
        ("Legal Heir Certificate", [
            "legal heir certificate", "legal heirs certificate", "சட்ட வாரிசுச் சான்றிதழ்", "வாரிசு சான்றிதழ்"
        ]),
        ("Residence / Nativity Certificate", [
            "residence certificate", "nativity certificate", "residential certificate", "இருப்பிடச் சான்றிதழ்", "குடியிருப்பு சான்றிதழ்"
        ]),
        ("Court Order", [
            "high court, madras", "high court of judicature", "court order", "judgment", "judgement",
            "c.r.p.no", "w.p.no", "நீதிமன்ற உத்தரவு", "நீதிமன்ற தீர்ப்பு"
        ]),
        ("Government Notice / Order", [
            "government order", "g.o.ms.no", "circular", "official notice", "அரசாணை", "அரசு உத்தரவு", "சுற்றறிக்கை"
        ]),
        ("Ration Card / Family Card", [
            "smart ration card", "ration card", "family card", "குடும்ப அட்டை", "ஸ்மார்ட் ரேஷன் கார்டு"
        ]),
        ("Aadhaar Document", [
            "unique identification authority of india", "aadhaar", "uidai", "ஆதார்"
        ]),
        ("Educational Certificate", [
            "mark sheet", "marks statement", "degree certificate", "higher secondary examination",
            "secondary school leaving certificate", "sslc", "hsc", "மதிப்பெண் சான்றிதழ்", "பல்கலைக்கழகம்"
        ]),
        ("Pension / Financial Assistance", [
            "pension payment order", "pensioner", "disability pension", "social security pension", "ஓய்வூதியம்"
        ]),
        ("Land / Property Document", [
            "patta / chitta", "patta copy", "chitta extract", "adangal", "sale deed",
            "பட்டா", "சிட்டா", "அடங்கல்", "கிரயப் பத்திரம்", "நில உரிமை"
        ]),
        ("Application Form", [
            "application form", "acknowledgement receipt", "விண்ணப்பப் படிவம்", "ஒப்புதல் ரசீது"
        ])
    ]

    for doc_type, phrases in exact_priority:
        for phrase in phrases:
            if phrase in lower:
                return (doc_type, 98.0, [phrase])

    # Pattern and keyword scoring
    keyword_scores = {
        "Land / Property Document": ["survey number", "patta no", "sub division", "extent", "village", "taluk", "சர்வே எண்", "பட்டா எண்", "நிலம்", "கிராமம்"],
        "Court Order": ["petitioner", "respondent", "plaintiff", "defendant", "disposed", "dismissed", "allowed", "மனுதாரர்", "வழக்கு"],
        "Government Notice / Order": ["department", "proceedings", "notification", "g.o.", "அரசாணை", "துறை", "அறிவிப்பு"],
        "Educational Certificate": ["examination", "register no", "marks obtained", "grade", "college", "school", "university", "தேர்வு"],
        "Income Certificate": ["annual income", "family income", "salary", "ஆண்டு வருமானம்", "வருமானம்"],
        "Community Certificate": ["community", "caste", "scheduled caste", "backward class", "சமூகம்", "சாதி"],
        "Legal Heir Certificate": ["legal heir", "deceased person", "surviving member", "சட்ட வாரிசு", "வாரிசு"],
        "Ration Card / Family Card": ["card holder", "fair price shop", "family head", "குடும்பத் தலைவர்", "ரேஷன் கடை"],
        "Application Form": ["applicant name", "signature", "application id", "விண்ணப்பம்", "விண்ணப்பதாரர்"]
    }

    best_type = "General Government Document"
    best_score = 0
    matched_kws = []

    for doc_type, kws in keyword_scores.items():
        score = sum(2 for kw in kws if kw in lower)
        if score > best_score:
            best_score = score
            best_type = doc_type
            matched_kws = [kw for kw in kws if kw in lower]

    if best_score >= 4:
        confidence = min(92.0, 60.0 + best_score * 3.5)
        return (best_type, confidence, matched_kws)

    # Universal Fallback
    return ("General Government Document", 50.0, ["general_government_record"])

def extract_general_fields(text, document_type):
    """Extract structured fields based on detected document type."""
    fields = {}
    flat_text = re.sub(r"\s+", " ", text).strip()

    if document_type == "First Graduate Certificate":
        applicant = NOT_DETECTED
        m = re.search(r"(?:this\s+is\s+to\s+certify\s+that|certify\s+that)\s+(?:smt\.?|smti\.?|ms\.?|miss|mr\.?|thiru\.?)?\s*([A-Za-z][A-Za-z .]{2,80}?)\s+(?:daughter|son|wife|husband|child)\s+of\b", flat_text, flags=re.IGNORECASE)
        if m:
            applicant = clean_extracted_value(m.group(1))
        if applicant == NOT_DETECTED:
            applicant = first_match_multiline([
                r"(?:Applicant Name|Name of Applicant|Student Name|செல்வி|திரு)\s*[:\-]?\s*(.+)",
                r"(?:விண்ணப்பதாரர் பெயர்|மாணவர் பெயர்)\s*[:\-]?\s*(.+)"
            ], text)
        fields["Applicant Name"] = applicant

        parent = NOT_DETECTED
        m = re.search(r"(?:daughter|son|child)\s+of\s+((?:thiru|mr\.?|mrs\.?|sri\.?)?\s*[A-Za-z][A-Za-z .]{2,80}?)\s+(?:residing|aged|from|having)\b", flat_text, flags=re.IGNORECASE)
        if m:
            parent = clean_extracted_value(m.group(1))
        if parent == NOT_DETECTED:
            parent = first_match_multiline([
                r"(?:Father(?:'s)? Name|Parent Name|Father Name)\s*[:\-]?\s*(.+)",
                r"(?:தந்தையின் பெயர்|பெற்றோர் பெயர்)\s*[:\-]?\s*(.+)"
            ], text)
        fields["Father's Name"] = parent

        fields["Certificate Number"] = first_match_multiline([
            r"(?:Certificate\s*No\.?|Certificate\s*Number|சான்றிதழ்\s*எண்)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)",
            r"\b(TN-[0-9]{10,20})\b"
        ], text)
        fields["District"] = first_match_multiline([
            r"(?:District|மாவட்டம்)\s*[:\-]?\s*([A-Za-z\u0B80-\u0BFF ]+)",
            r"([A-Za-z\u0B80-\u0BFF]+)\s+மாவட்டம்"
        ], text)

    elif document_type == "Income Certificate":
        fields["Applicant Name"] = first_match_multiline([
            r"(?:Applicant Name|Name of Applicant|Name)\s*[:\-]?\s*(.+)",
            r"(?:விண்ணப்பதாரர் பெயர்|பெயர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Father / Husband Name"] = first_match_multiline([
            r"(?:Father / Husband Name|Father(?:'s)? Name|Husband Name)\s*[:\-]?\s*(.+)",
            r"(?:தந்தை / கணவர் பெயர்|தந்தையின் பெயர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Annual Income"] = first_match_multiline([
            r"(?:Annual Family Income|Annual Income|Family Income|Income)\s*[:\-]?\s*(.+)",
            r"(?:குடும்ப ஆண்டு வருமானம்|ஆண்டு வருமானம்|வருமானம்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Certificate Number"] = first_match_multiline([
            r"(?:Income Certificate No\.?|Certificate No\.?|சான்றிதழ் எண்)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)",
            r"\b(TN-[0-9]{10,20})\b"
        ], text)
        fields["District"] = first_match_multiline([
            r"(?:District|மாவட்டம்)\s*[:\-]?\s*(.+)"
        ], text)

    elif document_type == "Community Certificate":
        fields["Applicant Name"] = first_match_multiline([
            r"(?:Name|Applicant Name|விண்ணப்பதாரர் பெயர்|பெயர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Community"] = first_match_multiline([
            r"(?:Community|சமூகம்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Caste"] = first_match_multiline([
            r"(?:Caste|Sub-caste|சாதி)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Certificate Number"] = first_match_multiline([
            r"(?:Community Certificate No\.?|Certificate No\.?|சான்றிதழ் எண்)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)",
            r"\b(TN-[0-9]{10,20})\b"
        ], text)

    elif document_type == "Land / Property Document":
        fields["Owner Name"] = first_match_multiline([
            r"(?:Owner Name|Name of Owner|Property Owner|உரிமையாளர் பெயர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Survey Number"] = first_match_multiline([
            r"(?:Survey Number|Survey No\.?|சர்வே எண்)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)"
        ], text)
        fields["Patta Number"] = first_match_multiline([
            r"(?:Patta Number|Patta No\.?|பட்டா எண்)\s*[:\-]?\s*([0-9\-_/]+)"
        ], text)
        fields["Village"] = first_match_multiline([
            r"(?:Village|கிராமம்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Taluk"] = first_match_multiline([
            r"(?:Taluk|வட்டம்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["District"] = first_match_multiline([
            r"(?:District|மாவட்டம்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Extent"] = first_match_multiline([
            r"(?:Extent|Land Extent|Area|பரப்பளவு)\s*[:\-]?\s*(.+)"
        ], text)

    elif document_type == "Birth Certificate":
        fields["Child Name"] = first_match_multiline([
            r"(?:Name of Child|Child Name|Name|குழந்தையின் பெயர்|பெயர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Father's Name"] = first_match_multiline([
            r"(?:Father(?:'s)? Name|தந்தையின் பெயர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Mother's Name"] = first_match_multiline([
            r"(?:Mother(?:'s)? Name|தாயின் பெயர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Date of Birth"] = first_match_multiline([
            r"(?:Date of Birth|DOB|பிறந்த தேதி)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Place of Birth"] = first_match_multiline([
            r"(?:Place of Birth|பிறந்த இடம்)\s*[:\-]?\s*(.+)"
        ], text)

    elif document_type == "Death Certificate":
        fields["Name of Deceased"] = first_match_multiline([
            r"(?:Name of Deceased|Deceased Person|Name|இறந்தவரின் பெயர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Date of Death"] = first_match_multiline([
            r"(?:Date of Death|DOD|இறந்த தேதி)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Place of Death"] = first_match_multiline([
            r"(?:Place of Death|இறந்த இடம்)\s*[:\-]?\s*(.+)"
        ], text)

    elif document_type == "Legal Heir Certificate":
        fields["Deceased Person"] = first_match_multiline([
            r"(?:Deceased Person|Name of Deceased|இறந்தவர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Legal Heirs"] = first_match_multiline([
            r"(?:Legal Heirs|Legal Heir|சட்ட வாரிசுகள்|வாரிசுகள்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Certificate Number"] = first_match_multiline([
            r"(?:Certificate No\.?|சான்றிதழ் எண்)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)"
        ], text)

    elif document_type == "Aadhaar Document":
        # Privacy protection: mask first 8 digits of Aadhaar number
        aadhaar_match = re.search(r"\b(\d{4})\s*(\d{4})\s*(\d{4})\b", text)
        if aadhaar_match:
            fields["Aadhaar Number"] = f"XXXX XXXX {aadhaar_match.group(3)}"
        else:
            fields["Aadhaar Number"] = "XXXX XXXX (Protected)"
        fields["Applicant Name"] = first_match_multiline([
            r"(?:Name|பெயர்)\s*[:\-]?\s*(.+)"
        ], text)

    elif document_type == "Ration Card / Family Card":
        fields["Ration Card Number"] = first_match_multiline([
            r"(?:Card No\.?|Ration Card No\.?|குடும்ப அட்டை எண்)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)"
        ], text)
        fields["Applicant Name"] = first_match_multiline([
            r"(?:Head of Family|Card Holder Name|குடும்பத் தலைவர்)\s*[:\-]?\s*(.+)"
        ], text)

    elif document_type == "Government Notice / Order":
        fields["Order / Notice Number"] = first_match_multiline([
            r"(?:G\.?O\.?\s*(?:Ms\.?)?\s*No\.?|Order No\.?|Notice No\.?|அரசாணை எண்|சுற்றறிக்கை)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Department"] = first_match_multiline([
            r"(?:Department|துறை)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Subject"] = first_match_multiline([
            r"(?:Subject|Sub|பொருள்)\s*[:\-]?\s*(.+)"
        ], text)

    else:
        # General Government Document Fallback
        fields["Name"] = first_match_multiline([
            r"(?:Name|Applicant Name|Beneficiary|பெயர்|விண்ணப்பதாரர்)\s*[:\-]?\s*(.+)"
        ], text)
        fields["Document / Reference Number"] = first_match_multiline([
            r"(?:Ref\s*No\.?|Notice\s*No\.?|Order\s*No\.?|Certificate\s*No\.?|எண்)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)"
        ], text)
        fields["Department"] = first_match_multiline([
            r"(?:Department|Office|Authority|துறை|அலுவலகம்)\s*[:\-]?\s*(.+)"
        ], text)

    # Dates extraction for all documents
    dates = extract_dates(text)
    if dates:
        fields["Dates"] = ", ".join(dates[:4])

    return fields

def build_fields_i18n(fields):
    return [
        {
            "label_en": key,
            "label_ta": FIELD_LABEL_TAMIL.get(key, key),
            "value": value
        }
        for key, value in (fields or {}).items()
        if value and value != NOT_DETECTED
    ]
