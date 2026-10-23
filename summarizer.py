"""
GovEase AI - Universal Government Document Summarization Engine
Bilingual (English & Tamil) summary generation, key entity extraction,
action points derivation, and condition analysis for all government documents.
"""

import re
from datetime import datetime

NOT_AVAILABLE_EN = "Not specified in document"
NOT_AVAILABLE_TA = "ஆவணத்தில் குறிப்பிடப்படவில்லை"

def extract_dates(text):
    """Extract standard date patterns from text."""
    date_patterns = [
        r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b",
    ]
    matches = []
    for pat in date_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            date_str = m.group(0).strip()
            if date_str not in matches:
                matches.append(date_str)
    return matches

def extract_reference_numbers(text):
    """Extract reference, certificate, G.O., order, case numbers."""
    ref_patterns = [
        r"(?:Certificate\s*No\.?|சான்றிதழ்\s*எண்)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)",
        r"(?:G\.?O\.?\s*(?:Ms\.?)?\s*No\.?|அரசாணை\s*எண்)\s*[:\-]?\s*([0-9\-_/]+)",
        r"(?:C\.?R\.?P\.?\s*(?:(?:Nos?\.?)|No)?|C\.?M\.?P\.?\s*No\.?|W\.?P\.?\s*No\.?|O\.?S\.?\s*No\.?|Case\s*No\.?|வழக்கு\s*எண்)\s*[:\.]?\s*([0-9\-_/&,\s]+of\s*\d{4}|\d+[/\-]\d+)",
        r"(?:Ref\s*(?:No\.?)?|Lr\s*No\.?|Notice\s*No\.?|Roc\s*No\.?|அறிவிப்பு\s*எண்)\s*[:\-]?\s*([A-Za-z0-9\-_/.]+)",
        r"\b(?:TN-[0-9]{10,20})\b",
    ]
    refs = []
    for pat in ref_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = m.group(1 if m.groups() else 0).strip()
            if val and len(val) >= 3 and val not in refs:
                refs.append(val)
    return refs[:5]

def extract_amounts(text):
    """Extract monetary amounts (salary, fees, income, compensation)."""
    amt_patterns = [
        r"(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d{2})?",
        r"[\d,]+\s*(?:Rupees|ரூபாய்)",
        r"(?:income|வருமானம்)\s*[:\-]?\s*(?:Rs\.?|₹)?\s*([\d,]+)",
    ]
    amounts = []
    for pat in amt_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            amt = m.group(0).strip()
            if amt not in amounts:
                amounts.append(amt)
    return amounts[:5]

def extract_authority(text):
    """Detect issuing government authority / department / court."""
    authorities_en = [
        ("High Court of Judicature at Madras / High Court", ["high court, madras", "high court of judicature", "high court"]),
        ("Revenue & Disaster Management Department", ["revenue department", "tahsildar", "revenue divisional officer", "taluk office"]),
        ("District Collectorate", ["district collector", "collectorate"]),
        ("Municipal Administration / Corporation", ["corporation", "municipality", "municipal commissioner"]),
        ("School / Higher Education Department", ["education department", "director of school education", "university"]),
        ("Social Welfare & Women Development", ["social welfare", "women development"]),
        ("Police / Judicial Magistrate Court", ["judicial magistrate", "police department", "sessions court"]),
    ]
    authorities_ta = [
        ("வருவாய்த் துறை / வட்டாட்சியர் அலுவலகம்", ["வருவாய்த் துறை", "வட்டாட்சியர்", "வட்டாட்சியர் அலுவலகம்", "வருவாய் கோட்டாட்சியர்"]),
        ("சென்னை உயர்நீதிமன்றம் / நீதிமன்றம்", ["உயர்நீதிமன்றம்", "சென்னை உயர்நீதிமன்றம்", "நீதிமன்றம்"]),
        ("மாவட்ட ஆட்சியர் அலுவலகம்", ["மாவட்ட ஆட்சியர்", "ஆட்சியரகம்"]),
        ("நகராட்சி / மாநகராட்சி நிர்வாகம்", ["மாநகராட்சி", "நகராட்சி"]),
        ("பள்ளிக் கல்வி / உயர்கல்வித் துறை", ["கல்வித் துறை", "பல்கலைக்கழகம்"]),
    ]
    
    text_lower = text.lower()
    auth_en = None
    auth_ta = None
    
    for title, kw_list in authorities_en:
        if any(kw in text_lower for kw in kw_list):
            auth_en = title
            break
            
    for title, kw_list in authorities_ta:
        if any(kw in text_lower for kw in kw_list):
            auth_ta = title
            break
            
    return (auth_en or "Government Authority / Competent Officer",
            auth_ta or "அரசுத் துறை / தகுதிவாய்ந்த அதிகாரி")

def summarize_document(text, document_type="General Government Document", fields=None):
    """
    Main Universal Summarization function.
    Returns:
    - quick_summary: { en: [...], ta: [...] }
    - detailed_summary: { en: "...", ta: "..." }
    - action_points: { en: [...], ta: [...] }
    - metadata: { authority_en, authority_ta, dates, refs, amounts }
    """
    if not text or len(text.strip()) < 15:
        empty_quick = {
            "en": ["Document text is unreadable or empty. Please upload a clearer copy or high-resolution scan."],
            "ta": ["ஆவண உரை தெளிவாக இல்லை அல்லது காலியாக உள்ளது. தெளிவான படம் அல்லது PDF பதிவேற்றவும்."]
        }
        empty_detailed = {
            "en": "The system could not extract readable government text from the uploaded file. Please verify that the image is well-lit, properly aligned, and in English or Tamil.",
            "ta": "பதிவேற்றப்பட்ட கோப்பிலிருந்து வாசிக்கக்கூடிய உரை கிடைக்கவில்லை. படம் தெளிவான ஒளியிலும் நேராகவும் இருப்பதை உறுதிசெய்யவும்."
        }
        empty_actions = {
            "en": ["Scan or photograph the document under good lighting.", "Ensure text, stamps, and certificate numbers are sharply focused.", "Re-upload the document for analysis."],
            "ta": ["நல்ல வெளிச்சத்தில் ஆவணத்தை படம் பிடிக்கவும்.", "முத்திரைகள், எண்கள் தெளிவாக இருப்பதை உறுதி செய்யவும்.", "மீண்டும் பதிவேற்றி பகுப்பாய்வு செய்யவும்."]
        }
        return {
            "quick_summary": empty_quick,
            "detailed_summary": empty_detailed,
            "action_points": empty_actions,
            "metadata": {
                "authority_en": NOT_AVAILABLE_EN,
                "authority_ta": NOT_AVAILABLE_TA,
                "dates": [],
                "refs": [],
                "amounts": [],
                "missing_fields": ["Readable text"]
            }
        }

    fields = fields or {}
    dates = extract_dates(text)
    refs = extract_reference_numbers(text)
    amounts = extract_amounts(text)
    auth_en, auth_ta = extract_authority(text)
    
    # Identify subject / person
    person_name = fields.get("Applicant Name") or fields.get("Name") or fields.get("Owner Name") or fields.get("Name of Deceased") or ""
    cert_no = fields.get("Certificate Number") or fields.get("Application Number") or fields.get("Case Number") or (refs[0] if refs else "")
    issue_date = fields.get("Order Date") or (dates[0] if dates else "")
    
    quick_en = []
    quick_ta = []
    detailed_en_paragraphs = []
    detailed_ta_paragraphs = []
    actions_en = []
    actions_ta = []
    missing_fields = []
    
    # Check for missing critical components
    if not cert_no or cert_no == "Not Detected":
        missing_fields.append("Certificate / Reference Number")
    if not dates:
        missing_fields.append("Issuance Date")

    # Document type specific logic
    doc_type_lower = document_type.lower()
    
    if "first graduate" in doc_type_lower:
        quick_en = [
            f"Official First Graduate Certificate verifying that the applicant is the first in their immediate family to pursue graduate degree education.",
            f"Issued to: {person_name or 'Candidate named in certificate'}." + (f" (Ref No: {cert_no})" if cert_no else ""),
            "Entitles the student to Tamil Nadu Government first-graduate tuition fee concessions and scholarship benefits in approved professional and arts/science colleges.",
            f"Issuing Department: {auth_en}." + (f" Issued on: {issue_date}" if issue_date else "")
        ]
        quick_ta = [
            f"விண்ணப்பதாரர் தனது குடும்பத்திலேயே முதல் தலைமுறை பட்டதாரி என்பதை உறுதிசெய்யும் அதிகாரப்பூர்வ சான்றிதழ்.",
            f"வழங்கப்பட்டவர்: {person_name or 'சான்றிதழில் உள்ள விண்ணப்பதாரர்'}." + (f" (எண்: {cert_no})" if cert_no else ""),
            "தமிழக அரசு உயர்கல்வி கல்விக்கட்டண சலுகை மற்றும் உதவித்தொகை பெற தகுதி அளிக்கிறது.",
            f"வழங்கிய துறை: {auth_ta}." + (f" வழங்கப்பட்ட தேதி: {issue_date}" if issue_date else "")
        ]
        detailed_en_paragraphs = [
            f"This document is an authenticated First Graduate Certificate issued under the authority of the Revenue and Disaster Management Department, Government of Tamil Nadu.",
            f"The certificate establishes that neither the applicant nor any members of their sibling/family hierarchy have previously completed higher education graduation. Therefore, the applicant qualifies for statutory educational benefits.",
            f"Key Details Recorded: Applicant: {person_name or 'Identified in document'}; Reference No: {cert_no or NOT_AVAILABLE_EN}; Issuing Authority: {auth_en}; Issued/Recorded Date: {issue_date or NOT_AVAILABLE_EN}."
        ]
        detailed_ta_paragraphs = [
            f"இந்த ஆவணம் தமிழ்நாடு அரசு வருவாய்த் துறையால் வழங்கப்பட்ட முதல் பட்டதாரி சான்றிதழ் ஆகும்.",
            f"விண்ணப்பதாரரின் குடும்பத்தில் வேறு எவரும் பட்டப்படிப்பு முடிக்கவில்லை என்பதை வருவாய்த் துறை சரிபார்த்து உறுதி செய்துள்ளது. இதன் மூலம் கல்லூரிகளில் அரசு கல்விக்கட்டண சலுகை பெற இயலும்.",
            f"பதிவுசெய்யப்பட்ட தகவல்கள்: விண்ணப்பதாரர்: {person_name or 'ஆவணத்தில் உள்ளது'}; சான்றிதழ் எண்: {cert_no or NOT_AVAILABLE_TA}; வழங்கிய அலுவலகம்: {auth_ta}; தேதி: {issue_date or NOT_AVAILABLE_TA}."
        ]
        actions_en = [
            "Submit an attested copy during college admissions counseling (TNEA / Medical / Arts & Science) for tuition fee waiver.",
            "Retain the digital certificate with QR code and Certificate Number securely for educational scholarships.",
            "Verify the digital authenticity on the official e-Sevai / TNeGA portal if requested by the institution."
        ]
        actions_ta = [
            "கல்லூரி சேர்க்கை மற்றும் கலந்தாய்வின் போது (TNEA/மருத்துவம்/கலை அறிவியல்) கல்விக் கட்டண விலக்கு பெற இந்த சான்றிதழை சமர்ப்பிக்கவும்.",
            "சான்றிதழ் எண் மற்றும் டிஜிட்டல் நகலை உயர்கல்வி உதவித்தொகை விண்ணப்பங்களுக்கு பாதுகாப்பாக வைக்கவும்.",
            "தேவைப்பட்டால் தமிழ்நாடு மின்சேவை இணையதளத்தில் இதன் நம்பகத்தன்மையை சரிபார்க்கலாம்."
        ]

    elif "court" in doc_type_lower or "circular" in doc_type_lower:
        quick_en = [
            f"Official Judicial Order / Circular issued by {auth_en}.",
            f"Subject & Reference: {('Ref: ' + cert_no) if cert_no else 'Legal directives and registry instructions.'}",
            "Outlines binding legal directives, court proceedings, case adjudication, or statutory administrative guidelines.",
            f"Recorded Order Date: {issue_date or (dates[0] if dates else NOT_AVAILABLE_EN)}."
        ]
        quick_ta = [
            f"{auth_ta} மூலம் வெளியிடப்பட்ட அதிகாரப்பூர்வ நீதிமன்ற உத்தரவு / சுற்றறிக்கை.",
            f"வழக்கு / குறிப்பு எண்: {cert_no or 'ஆவணத்தில் குறிப்பிடப்பட்டுள்ளது'}.",
            "நீதிமன்ற நடவடிக்கைகள், வழக்கு முடிவு, காலதாமத மன்னிப்பு அல்லது சட்ட வழிகாட்டுதல்களை தெரிவிக்கிறது.",
            f"உத்தரவு நாள்: {issue_date or (dates[0] if dates else NOT_AVAILABLE_TA)}."
        ]
        detailed_en_paragraphs = [
            f"This document represents a judicial order or official procedural circular issued by {auth_en}.",
            "It conveys formal directions issued in accordance with judicial proceedings. Citizens and advocates must comply with the prescribed timelines, appearances, and statutory conditions mentioned.",
            f"Important Reference Identifiers: {', '.join(refs) if refs else 'Refer to extracted text for specific case numbers'}."
        ]
        detailed_ta_paragraphs = [
            f"இந்த ஆவணம் {auth_ta} மூலம் பிறப்பிக்கப்பட்ட நீதித்துறை உத்தரவு அல்லது சுற்றறிக்கை ஆகும்.",
            "நீதிமன்ற நடவடிக்கைகளின்படி பிறப்பிக்கப்பட்ட உத்தரவுகள் மற்றும் சட்ட விதிமுறைகளை இந்த ஆவணம் விளக்குகிறது. இதில் குறிப்பிடப்பட்டுள்ள வழிகாட்டுதல்கள் மற்றும் காலக்கெடுவை பின்பற்ற வேண்டும்.",
            f"வழக்கு / சுற்றறிக்கை எண்கள்: {', '.join(refs) if refs else 'ஆவண உரையை பார்க்கவும்'}."
        ]
        actions_en = [
            "Review the specific case directions, interim orders, and deadlines mentioned in the order.",
            "Consult with your legal counsel regarding compliance or filing of required petitions.",
            "Preserve certified copies for court record submissions and official communications."
        ]
        actions_ta = [
            "உத்தரவில் குறிப்பிடப்பட்டுள்ள குறிப்பிட்ட வழக்கு வழிகாட்டுதல்கள் மற்றும் காலக்கெடுவை கவனமாக சரிபார்க்கவும்.",
            "தேவையான மனுக்கள் தாக்கல் செய்வது தொடர்பாக உங்கள் வழக்கறிஞருடன் கலந்தாலோசிக்கவும்.",
            "நீதிமன்ற ஆவணங்களுக்காக சான்றளிக்கப்பட்ட நகலை பாதுகாப்பாக சேமித்து வைக்கவும்."
        ]

    elif "income" in doc_type_lower:
        quick_en = [
            f"Government Income Certificate verifying annual family income for {person_name or 'the applicant'}.",
            f"Assessed Annual Income: {fields.get('Annual Income') or (amounts[0] if amounts else 'Recorded in document')}.",
            "Crucial proof of income required for government welfare schemes, educational fee concessions, and scholarships.",
            f"Certificate Reference: {cert_no or 'Recorded on certificate'} | Issued: {issue_date or 'Recent'}"
        ]
        quick_ta = [
            f"{person_name or 'விண்ணப்பதாரரின்'} குடும்ப ஆண்டு வருமானத்தை உறுதிப்படுத்தும் அரசு வருமானச் சான்றிதழ்.",
            f"ஆண்டு வருமானம்: {fields.get('Annual Income') or (amounts[0] if amounts else 'ஆவணத்தில் குறிப்பிடப்பட்டுள்ளது')}.",
            "அரசு நலத்திட்டங்கள், கல்விக் கட்டணச் சலுகைகள் மற்றும் உதவித்தொகைகளுக்கு அத்தியாவசியமான ஆவணம்.",
            f"சான்றிதழ் எண்: {cert_no or 'ஆவணத்தில் உள்ளது'} | வழங்கப்பட்ட நாள்: {issue_date or 'குறிப்பிடப்பட்டுள்ளது'}"
        ]
        detailed_en_paragraphs = [
            f"This Income Certificate was issued by the Revenue Department ({auth_en}) after official revenue verification of the applicant's household earnings.",
            f"Annual income certificates typically remain valid for one financial year from the date of issue. Ensure validity before submitting for new academic or scheme cycles.",
            f"Key Extracted Details: Beneficiary: {person_name or NOT_AVAILABLE_EN}; Income: {fields.get('Annual Income') or NOT_AVAILABLE_EN}; Taluk/District: {fields.get('District') or 'State Revenue Office'}."
        ]
        detailed_ta_paragraphs = [
            f"இந்த வருமானச் சான்றிதழ் வருவாய்த் துறையால் ({auth_ta}) குடும்ப வருமானத்தை சரிபார்த்து வழங்கப்பட்டுள்ளது.",
            "பொதுவாக வருமானச் சான்றிதழ் வழங்கப்பட்ட தேதியிலிருந்து ஒரு நிதி ஆண்டுக்கு மட்டுமே செல்லுபடியாகும். தற்போதைய நிதி ஆண்டுக்கானதா என்பதை சரிபார்க்கவும்.",
            f"முக்கிய தகவல்கள்: பயனாளி: {person_name or NOT_AVAILABLE_TA}; வருமானம்: {fields.get('Annual Income') or NOT_AVAILABLE_TA}; மாவட்டம்: {fields.get('District') or 'வருவாய் அலுவலகம்'}."
        ]
        actions_en = [
            "Verify that the certificate is within its valid period (typically 1 year from issue date).",
            "Submit for scholarship, reservation, or government welfare scheme applications.",
            "If expired, apply for renewal through your local e-Sevai / revenue counter."
        ]
        actions_ta = [
            "சான்றிதழ் செல்லுபடியாகும் காலத்திற்குள் (வழக்கமாக 1 ஆண்டு) உள்ளதா என்பதை உறுதி செய்யவும்.",
            "கல்வி உதவித்தொகை மற்றும் அரசு நலத்திட்டங்களுக்கு விண்ணப்பிக்க பயன்படுத்தவும்.",
            "காலாவதியாகியிருந்தால் உள்ளூர் மின்சேவை மையம் மூலம் புதுப்பிக்க விண்ணப்பிக்கவும்."
        ]

    elif "community" in doc_type_lower or "caste" in doc_type_lower:
        quick_en = [
            f"Government Community / Caste Certificate for {person_name or 'the applicant'}.",
            f"Designated Community: {fields.get('Community') or fields.get('Caste') or 'Stated in certificate'}.",
            "Serves as permanent statutory evidence of social category for education, government employment reservations, and welfare eligibility.",
            f"Certificate Reference: {cert_no or NOT_AVAILABLE_EN}."
        ]
        quick_ta = [
            f"{person_name or 'விண்ணப்பதாரருக்கான'} அரசு சமூக / சாதிச் சான்றிதழ்.",
            f"சமூகம் / பிரிவு: {fields.get('Community') or fields.get('Caste') or 'சான்றிதழில் குறிப்பிடப்பட்டுள்ளது'}.",
            "கல்வி, அரசு வேலைவாய்ப்பு இடஒதுக்கீடு மற்றும் நலத்திட்ட தகுதிக்கான நிரந்தர சட்டபூர்வ ஆவணம்.",
            f"சான்றிதழ் எண்: {cert_no or NOT_AVAILABLE_TA}."
        ]
        detailed_en_paragraphs = [
            f"This is an official Community Certificate issued by {auth_en}.",
            "Community certificates do not expire unless cancelled by competent authorities. It confers constitutional reservation benefits in education, competitive exams, and government job quotas.",
            f"Details: Person: {person_name or 'Subject'}; Community/Caste: {fields.get('Community', '')} {fields.get('Caste', '')}; Reference: {cert_no}."
        ]
        detailed_ta_paragraphs = [
            f"இந்த ஆவணம் {auth_ta} மூலம் வழங்கப்பட்ட சமூகச் சான்றிதழ் ஆகும்.",
            "சமூகச் சான்றிதழ்களுக்கு காலாவதி தேதி இல்லை; நிரந்தரமாக செல்லுபடியாகும். கல்வி மற்றும் அரசு பணிகளில் இடஒதுக்கீடு பெற இது மிக முக்கிய ஆவணம்.",
            f"தகவல்கள்: பெயர்: {person_name or 'பயனாளி'}; சமூகம்: {fields.get('Community', '')} {fields.get('Caste', '')}; எண்: {cert_no}."
        ]
        actions_en = [
            "Store safely as a permanent credential for school/college admissions and competitive examinations.",
            "Ensure the name spelling matches your Aadhaar and secondary school mark sheets.",
            "Use certificate number to link with DigiLocker or scholarship portals."
        ]
        actions_ta = [
            "கல்வி மற்றும் அரசுப் பணி தேர்வுகளுக்காக நிரந்தர ஆவணமாக பாதுகாப்பாக வைக்கவும்.",
            "பெயர் எழுத்துப்பிழை ஆதார் மற்றும் பள்ளி சான்றிதழ்களுடன் பொருந்துகிறதா என்பதை சரிபார்க்கவும்.",
            "டிஜிலாக்கர் அல்லது உதவித்தொகை இணையதளங்களில் சான்றிதழ் எண்ணை இணைக்கவும்."
        ]

    elif "land" in doc_type_lower or "property" in doc_type_lower:
        quick_en = [
            f"Official Land Record / Property Ownership document for {fields.get('Owner Name') or person_name or 'the title holder'}.",
            f"Survey / Patta Details: Survey No: {fields.get('Survey Number', 'Recorded')}; Patta No: {fields.get('Patta Number', 'Recorded')}.",
            f"Location: {fields.get('Village', '')}, {fields.get('Taluk', '')}, {fields.get('District', '')}.",
            f"Land Extent / Area: {fields.get('Extent', 'As per revenue register')}."
        ]
        quick_ta = [
            f"{fields.get('Owner Name') or person_name or 'உரிமையாளருக்கான'} நில ஆவணம் / பட்டா / சொத்து பதிவு ஆவணம்.",
            f"சர்வே / பட்டா விவரம்: சர்வே எண்: {fields.get('Survey Number', 'ஆவணத்தில் உள்ளது')}; பட்டா எண்: {fields.get('Patta Number', 'ஆவணத்தில் உள்ளது')}.",
            f"இடம்: {fields.get('Village', '')}, {fields.get('Taluk', '')}, {fields.get('District', '')}.",
            f"நிலப்பரப்பு: {fields.get('Extent', 'வருவாய் பதிவேட்டின்படி')}."
        ]
        detailed_en_paragraphs = [
            f"This document evidences land ownership, tenancy, or revenue registration maintained by {auth_en}.",
            "It records statutory land boundaries, survey subdivision details, village revenue accounts, and registered ownership.",
            "Ensure survey numbers and subdivision boundaries correspond accurately with physical land possession and registration deeds."
        ]
        detailed_ta_paragraphs = [
            f"இந்த ஆவணம் {auth_ta} பராமரிக்கும் நில உரிமை அல்லது பட்டா பதிவு ஆகும்.",
            "நிலத்தின் சர்வே எண், உட்பிரிவு, கிராமம் மற்றும் உரிமையாளர் பெயர் குறித்த வருவாய் துறை விவரங்களை இது கொண்டுள்ளது.",
            "சர்வே எண்களும் நிலத்தின் பரப்பளவும் உங்கள் நேரடி பயன்பாட்டுடனும் பத்திரத்துடனும் சரியாக ஒத்துப்போகிறதா என்பதை உறுதி செய்யவும்."
        ]
        actions_en = [
            "Cross-verify the survey number and patta number on the official Tamil Nadu AnyWhere AnyTime e-Services (eservices.tn.gov.in) portal.",
            "Check for recent encumbrance records (EC) before any sale, lease, or agricultural loan application.",
            "Preserve alongside registered sale deed and tax receipts."
        ]
        actions_ta = [
            "தமிழ்நாடு நில நிர்வாக இணையதளத்தில் (eservices.tn.gov.in) பட்டா மற்றும் சர்வே விவரங்களை சரிபார்க்கவும்.",
            "விற்பனை அல்லது விவசாய கடன் பெறுவதற்கு முன் வில்லங்கச் சான்றிதழை (EC) சரிபார்க்கவும்.",
            "கிரயப் பத்திரம் மற்றும் நில வரி ரசீதுகளுடன் இந்த ஆவணத்தை இணைத்து பாதுகாக்கவும்."
        ]

    else:
        # General Government Document Fallback
        quick_en = [
            f"Official Government Document classified under {document_type}.",
            f"Identified Authority / Department: {auth_en}.",
            f"Key Reference: {cert_no or (refs[0] if refs else 'Recorded in text')}." + (f" | Dated: {issue_date}" if issue_date else ""),
            f"Subject / Individual: {person_name or 'Stated within document content'}."
        ]
        quick_ta = [
            f"{document_type} பிரிவின் கீழ் வகைப்படுத்தப்பட்ட அதிகாரப்பூர்வ அரசு ஆவணம்.",
            f"வழங்கிய துறை / அலுவலகம்: {auth_ta}.",
            f"குறிப்பு / ஆவண எண்: {cert_no or (refs[0] if refs else 'ஆவணத்தில் உள்ளது')}." + (f" | நாள்: {issue_date}" if issue_date else ""),
            f"தொடர்புடைய நபர் / பொருள்: {person_name or 'ஆவணத்தில் குறிப்பிடப்பட்டுள்ளது'}."
        ]
        detailed_en_paragraphs = [
            f"This is an official government record or official correspondence issued by {auth_en}.",
            f"The document provides structured administrative declarations, notifications, or citizen entitlements. All dates, reference numbers, and designated conditions should be preserved.",
            f"Referenced Information: {len(dates)} dates identified, {len(refs)} reference identifiers detected, and monetary amounts noted: {', '.join(amounts) if amounts else 'None stated'}."
        ]
        detailed_ta_paragraphs = [
            f"இந்த ஆவணம் {auth_ta} மூலம் வெளியிடப்பட்ட அரசுப் பதிவு அல்லது அதிகாரப்பூர்வ அறிவிப்பு ஆகும்.",
            "இதில் குறிப்பிடப்பட்டுள்ள அரசு விதிமுறைகள், சான்றளிப்புகள் மற்றும் உத்தரவுகள் குடிமக்கள் அல்லது அலுவலக பயன்பாட்டிற்குரியவை.",
            f"கண்டறியப்பட்ட தகவல்கள்: {len(dates)} தேதிகள், {len(refs)} குறிப்பு எண்கள், மற்றும் தொகை விவரங்கள்: {', '.join(amounts) if amounts else 'குறிப்பிடப்படவில்லை'}."
        ]
        actions_en = [
            "Check all critical reference numbers and dates against your existing official records.",
            "Verify any application deadlines, response timeframes, or fee requirements mentioned.",
            "Store a digital copy securely and present the original when requested by authorities."
        ]
        actions_ta = [
            "ஆவணத்தில் உள்ள எண்கள் மற்றும் தேதிகள் உங்கள் பிற அரசு ஆவணங்களுடன் சரியாக உள்ளதா என சரிபார்க்கவும்.",
            "விண்ணப்ப காலக்கெடு அல்லது கட்டண தேவைகள் ஏதேனும் உள்ளதா என்பதை கவனிக்கவும்.",
            "டிஜிட்டல் நகலை பாதுகாப்பாக சேமித்து, அதிகாரிகள் கேட்கும்போது அசல் ஆவணத்தை சமர்ப்பிக்கவும்."
        ]

    detailed_en = "\n\n".join(detailed_en_paragraphs)
    detailed_ta = "\n\n".join(detailed_ta_paragraphs)

    return {
        "quick_summary": {
            "en": quick_en,
            "ta": quick_ta
        },
        "detailed_summary": {
            "en": detailed_en,
            "ta": detailed_ta
        },
        "action_points": {
            "en": actions_en,
            "ta": actions_ta
        },
        "metadata": {
            "authority_en": auth_en,
            "authority_ta": auth_ta,
            "dates": dates,
            "refs": refs,
            "amounts": amounts,
            "missing_fields": missing_fields
        }
    }
