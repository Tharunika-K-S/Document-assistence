"""
GovEase AI - Court Order & Legal Notice Parsing Module
Extracts case numbers, court names, judges, parties, outcomes, and directives.
"""

import re

NOT_DETECTED = "Not Detected"

def clean_extracted_value(value):
    if not value:
        return NOT_DETECTED
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^[:\-\s]+", "", value)
    value = re.sub(r"[:\-\s]+$", "", value)
    return value if value else NOT_DETECTED

def court_lines(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            line = re.sub(r"\s+", " ", line)
            lines.append(line)
    return lines

def first_match(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            val = clean_extracted_value(match.group(1))
            if val != NOT_DETECTED:
                return val
    return NOT_DETECTED

def find_case_number(text):
    patterns = [
        r"\b(?:Case\s*No\.?|Case\s*Number)\s*[:\-]?\s*([A-Za-z0-9./()\-]+(?:\s+of\s+\d{4})?)",
        r"\b(?:C\.?R\.?P\.?\s*(?:Nos?\.?)?|C\.?M\.?P\.?\s*No\.?|W\.?\s*P\.?|Crl\.?\s*O\.?\s*P\.?|O\.?\s*S\.?|S\.?\s*A\.?|C\.?\s*M\.?\s*A\.?|A\.?\s*S\.?|C\.?\s*C\.?)\s*(?:No\.?|Number|Nos\.?)?\s*([A-Za-z0-9./()\-&,\s]+(?:\s+of\s+\d{4})?)",
        r"(?:வழக்கு\s*எண்)\s*[:\-]?\s*([A-Za-z0-9./()\-]+(?:\s+of\s+\d{4})?)"
    ]
    result = first_match(patterns, text)
    if result != NOT_DETECTED:
        result = re.sub(r"\s+(?=(Petitioner|Respondent|Plaintiff|Defendant)\b).*", "", result, flags=re.IGNORECASE)
        return result.strip()
    return NOT_DETECTED

def find_labeled_person(text, labels):
    lines = court_lines(text)
    for line in lines:
        for label in labels:
            pattern = r"^\s*" + re.escape(label) + r"\s*[:\-]\s*(.+)$"
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                value = clean_extracted_value(match.group(1))
                if value and value != NOT_DETECTED:
                    return value
    return NOT_DETECTED

def find_court_name(text):
    lines = court_lines(text)
    court_terms = [
        "high court", "district court", "supreme court", "sessions court",
        "civil court", "family court", "magistrate court", "judicial magistrate",
        "court of", "நீதிமன்றம்", "உயர்நீதிமன்றம்", "மாவட்ட நீதிமன்றம்"
    ]
    for line in lines[:40]:
        lower = line.lower()
        if any(term in lower for term in court_terms):
            return line[:250]
    return NOT_DETECTED

def find_judge_name(text):
    lines = court_lines(text)
    judge_labels = [
        "judge", "justice", "presiding officer", "hon'ble judge", "honble judge",
        "honourable judge", "நீதிபதி", "மாண்புமிகு நீதிபதி"
    ]
    for line in lines[:50]:
        lower = line.lower()
        if any(lbl in lower for lbl in judge_labels):
            if len(line) <= 250:
                return line
    return NOT_DETECTED

def find_order_date(text):
    patterns = [
        r"(?:Date\s+of\s+Order|Order\s+Date|Date\s+of\s+Judgment|DATED)\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
        r"(?:DATED\s+THIS\s+THE)\s*(.+)",
        r"(?:உத்தரவு\s+தேதி|தீர்ப்பு\s+தேதி)\s*[:\-]\s*(.+)"
    ]
    res = first_match(patterns, text)
    if res != NOT_DETECTED:
        return res[:50]
    date_matches = re.findall(r"\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b", text)
    if date_matches:
        return date_matches[0]
    return NOT_DETECTED

def find_petitioner(text):
    return find_labeled_person(
        text,
        ["Petitioner", "Petitioner Name", "Petitioners", "Plaintiff", "Appellant", "மனுதாரர்"]
    )

def find_respondent(text):
    return find_labeled_person(
        text,
        ["Respondent", "Respondent Name", "Respondents", "Defendant", "எதிர்மனுதாரர்"]
    )

def detect_court_outcome(text):
    lines = court_lines(text)
    outcomes = [
        ("Partly Allowed", ["partly allowed", "partially allowed", "பகுதியளவில் அனுமதிக்கப்படுகிறது"]),
        ("Dismissed", ["petition is dismissed", "petition dismissed", "dismissed", "தள்ளுபடி செய்யப்படுகிறது", "தள்ளுபடி"]),
        ("Allowed", ["petition is allowed", "petition allowed", "allowed", "அனுமதிக்கப்படுகிறது", "அனுமதிக்கப்பட்டது"]),
        ("Disposed", ["petition is disposed", "petition disposed", "disposed of", "disposed", "வழக்கு முடிக்கப்பட்டது", "முடிவு"]),
        ("Withdrawn", ["petition withdrawn", "withdrawn", "திரும்பப் பெறப்பட்டது"]),
        ("Rejected", ["rejected", "rejection", "நிராகரிக்கப்படுகிறது"]),
        ("Granted", ["application is granted", "granted", "அனுமதி"]),
        ("Adjourned", ["adjourned", "ஒத்திவைக்கப்படுகிறது"]),
        ("Directions Issued", ["directions issued", "circular issued", "guidelines issued", "சுற்றறிக்கை"]),
    ]
    search_lines = lines[-40:] if len(lines) > 40 else lines
    for outcome_name, keywords in outcomes:
        for line in reversed(search_lines):
            lower = line.lower()
            for kw in keywords:
                if kw in lower:
                    return {"name": outcome_name, "source_line": line}
    return {"name": NOT_DETECTED, "source_line": ""}

def extract_order_points(text):
    lines = court_lines(text)
    direction_keywords = [
        "directed", "hereby directed", "ordered", "shall be", "shall", "restrained",
        "granted", "dismissed", "allowed", "disposed", "stay", "injunction", "directions issued",
        "உத்தரவிடப்படுகிறது", "அனுமதிக்கப்படுகிறது", "தள்ளுபடி", "சுற்றறிக்கை"
    ]
    points = []
    candidate_lines = lines[-80:] if len(lines) > 80 else lines
    for line in candidate_lines:
        lower = line.lower()
        if len(line) < 15:
            continue
        if any(kw in lower for kw in direction_keywords):
            if line not in points:
                points.append(line[:400])
        if len(points) >= 5:
            break
    return points

def simplify_court_order(text):
    case_no = find_case_number(text)
    court = find_court_name(text)
    judge = find_judge_name(text)
    order_date = find_order_date(text)
    petitioner = find_petitioner(text)
    respondent = find_respondent(text)
    outcome_dict = detect_court_outcome(text)
    outcome = outcome_dict["name"]
    important_points = extract_order_points(text)

    en_sum = [
        "Official Judicial Order / Proceeding detected.",
        f"Case / Proceeding Number: {case_no}.",
        f"Forum: {court} | Order Date: {order_date}.",
        f"Adjudicated Outcome / Directives: {outcome}."
    ]
    ta_sum = [
        "நீதிமன்ற உத்தரவு / நீதித்துறை நடவடிக்கை ஆவணம்.",
        f"வழக்கு / குறிப்பு எண்: {case_no}.",
        f"நீதிமன்றம்: {court} | தேதி: {order_date}.",
        f"வழக்கின் முடிவு / உத்தரவு: {outcome}."
    ]

    return {
        "case_number": case_no,
        "court_name": court,
        "judge": judge,
        "order_date": order_date,
        "petitioner": petitioner,
        "respondent": respondent,
        "outcome": outcome,
        "important_points": important_points,
        "english_summary": en_sum,
        "tamil_summary": ta_sum
    }
