"""
GovEase AI - Bilingual Voice Assistant Backend Module
Handles:
1. Speech-to-Text Q&A processing for Document Understanding & Meaning.
2. Voice-driven Government Scheme Q&A and Recommendation.
3. Natural bilingual answer generation in Tamil & English.
4. Robust fallback when microphone or context is unavailable.
"""

import os
import re
import json
import logging
from flask import Blueprint, request, jsonify, session

logger = logging.getLogger("govease.voice")
logger.setLevel(logging.INFO)

voice_bp = Blueprint("voice_assistant", __name__, url_prefix="/api/voice")

_CURRENT_RESULT_GETTER = None

def register_result_getter(getter_fn):
    """Registers app.py get_current_result callback."""
    global _CURRENT_RESULT_GETTER
    _CURRENT_RESULT_GETTER = getter_fn


def is_tamil_text(text):
    """Returns True if string contains Tamil Unicode characters."""
    if not text:
        return False
    return bool(re.search(r'[\u0b80-\u0bff]', text))


# ============================================================
# DOCUMENT QUESTION ANSWERING ENGINE
# ============================================================

def answer_document_question(question, doc_result, lang="en"):
    """
    Analyzes user question regarding the uploaded document and generates a concise,
    natural spoken-language answer in Tamil or English.
    """
    if not doc_result:
        if lang == "ta" or is_tamil_text(question):
            return {
                "answer": "தற்போது செயலில் ஆவணம் எதுவும் இல்லை. தயவுசெய்து முதலில் ஒரு ஆவணத்தை பதிவேற்றவும்.",
                "lang": "ta"
            }
        return {
            "answer": "No document is currently active. Please upload or view a document first to ask questions about it.",
            "lang": "en"
        }

    q_lower = (question or "").lower().strip()
    is_ta = (lang == "ta") or is_tamil_text(question)

    doc_type = doc_result.get("document_type", "General Government Document")
    doc_type_ta = doc_result.get("document_type_ta", "பொது அரசு ஆவணம்")
    summary = doc_result.get("summary", {}) or {}
    quick_summary = summary.get("quick_summary", {}) or {}
    action_points = summary.get("action_points", {}) or {}
    detailed_summary = summary.get("detailed_summary", {}) or {}
    fields = doc_result.get("fields", {}) or {}
    court_order = doc_result.get("court_order")
    ocr_text = doc_result.get("text", "")

    # 1. QUALITY / CONFIDENCE INTENT (தரம் / துல்லியம்)
    if any(k in q_lower for k in ["quality", "confidence", "readable", "clear", "accuracy"]) or \
       any(k in question for k in ["தரம்", "நம்பகத்தன்மை", "தெளிவு", "துல்லியம்"]):
        qual = doc_result.get("quality", "Moderate")
        conf = doc_result.get("confidence", 0)
        if is_ta:
            return {"answer": f"ஆவணப் பகுப்பாய்வு தரம்: {qual}, துல்லியம்: {conf}%", "lang": "ta", "intent": "quality"}
        return {"answer": f"Document OCR analysis quality is {qual} with a confidence score of {conf}%.", "lang": "en", "intent": "quality"}

    # 2. DATE INTENT (தேதி)
    if any(k in q_lower for k in ["date", "when", "issue date", "order date", "issued"]) or \
       any(k in question for k in ["தேதி", "நாள்", "எப்போது"]):
        found_date = fields.get("Date") or fields.get("Order Date") or fields.get("Issue Date")
        if not found_date and court_order:
            found_date = court_order.get("order_date")
        if not found_date:
            dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', ocr_text)
            if dates:
                found_date = dates[0]

        if found_date:
            if is_ta:
                return {"answer": f"ஆவணத்தில் குறிப்பிடப்பட்டுள்ள தேதி: {found_date}.", "lang": "ta", "intent": "date"}
            return {"answer": f"The date mentioned in this document is {found_date}.", "lang": "en", "intent": "date"}
        else:
            if is_ta:
                return {"answer": "ஆவணத்தில் குறிப்பிட்ட தேதி தெளிவாகக் கண்டறியப்படவில்லை.", "lang": "ta", "intent": "date"}
            return {"answer": "A specific date was not clearly detected in the extracted text.", "lang": "en", "intent": "date"}

    # 3. PARTIES / NAMES INTENT (பெயர் / மனுதாரர் / விண்ணப்பதாரர்)
    if any(k in q_lower for k in ["who", "name", "applicant", "petitioner", "respondent", "party", "person"]) or \
       any(k in question for k in ["யார்", "பெயர்", "விண்ணப்பதாரர்", "மனுதாரர்", "எதிர்மனுதாரர்"]):
        petitioner = fields.get("Petitioner") or (court_order.get("petitioner") if court_order else None)
        respondent = fields.get("Respondent") or (court_order.get("respondent") if court_order else None)
        name = fields.get("Applicant Name") or fields.get("Name")

        if petitioner or respondent:
            if is_ta:
                parts = []
                if petitioner: parts.append(f"மனுதாரர்: {petitioner}")
                if respondent: parts.append(f"எதிர்மனுதாரர்: {respondent}")
                return {"answer": ", ".join(parts), "lang": "ta", "intent": "parties"}
            else:
                parts = []
                if petitioner: parts.append(f"Petitioner is {petitioner}")
                if respondent: parts.append(f"Respondent is {respondent}")
                return {"answer": ", ".join(parts), "lang": "en", "intent": "parties"}
        elif name:
            if is_ta:
                return {"answer": f"ஆவணத்தில் உள்ள பெயர்: {name}.", "lang": "ta", "intent": "name"}
            return {"answer": f"The applicant or holder name found is {name}.", "lang": "en", "intent": "name"}

    # 4. COURT ORDER OUTCOME / VERDICT INTENT (தீர்ப்பு / நீதிமன்ற முடிவு)
    if any(k in q_lower for k in ["court", "outcome", "verdict", "judgment", "ruling", "decision"]) or \
       any(k in question for k in ["தீர்ப்பு", "முடிவு", "நீதிமன்ற தீர்ப்பு"]):
        if court_order:
            outcome = court_order.get("outcome", "")
            case_no = court_order.get("case_number", "")
            court_name = court_order.get("court_name", "")
            if is_ta:
                ans = f"வழக்கு எண் {case_no}. நீதிமன்ற முடிவு: {outcome}." if outcome else f"நீதிமன்ற ஆவணம் {court_name}."
                return {"answer": ans, "lang": "ta", "intent": "court"}
            else:
                ans = f"Case Number: {case_no}. Outcome: {outcome}." if outcome else f"Court order issued by {court_name}."
                return {"answer": ans, "lang": "en", "intent": "court"}

    # 5. SUMMARY INTENT (சுருக்கம்)
    if any(k in q_lower for k in ["summary", "about", "explain", "overview", "what is this", "brief"]) or \
       any(k in question for k in ["சுருக்கம்", "பற்றி", "விளக்க", "என்ன ஆவணம்", "விளக்கு"]):
        if is_ta:
            bullets = quick_summary.get("ta", [])
            if bullets:
                ans = f"இந்த ஆவணம் {doc_type_ta} வகையைச் சேர்ந்தது. முக்கிய குறிப்புகள்: " + "; ".join(bullets[:3])
            elif detailed_summary.get("ta"):
                ans = detailed_summary.get("ta")[:280] + "..."
            else:
                ans = f"இந்த ஆவணம் {doc_type_ta} என அடையாளம் காணப்பட்டுள்ளது."
            return {"answer": ans, "lang": "ta", "intent": "summary"}
        else:
            bullets = quick_summary.get("en", [])
            if bullets:
                ans = f"This document is a {doc_type}. Key points: " + "; ".join(bullets[:3])
            elif detailed_summary.get("en"):
                ans = detailed_summary.get("en")[:280] + "..."
            else:
                ans = f"This document has been identified as a {doc_type}."
            return {"answer": ans, "lang": "en", "intent": "summary"}

    # 6. ACTION POINTS INTENT (அடுத்த கட்ட நடவடிக்கை)
    if any(k in q_lower for k in ["action", "what next", "what should i do", "steps", "how to proceed", "process"]) or \
       any(k in question for k in ["நடவடிக்கை", "அடுத்த", "என்ன செய்ய", "வழிமுறை"]):
        if is_ta:
            actions = action_points.get("ta", [])
            if actions:
                ans = "உங்கள் அடுத்த கட்ட நடவடிக்கைகள்: " + "; ".join([f"{i+1}. {a}" for i, a in enumerate(actions[:3])])
            else:
                ans = "அடுத்த கட்ட நடவடிக்கை: ஆவணத்தின் உண்மைத்தன்மையை அதிகாரிகளிடம் சரிபார்த்துக்கொள்ளவும்."
            return {"answer": ans, "lang": "ta", "intent": "actions"}
        else:
            actions = action_points.get("en", [])
            if actions:
                ans = "Your next recommended action steps are: " + "; ".join([f"{i+1}. {a}" for i, a in enumerate(actions[:3])])
            else:
                ans = "Next action: Please verify this document with the issuing department."
            return {"answer": ans, "lang": "en", "intent": "actions"}

    # 7. GENERAL KEYWORD / SENTENCE SEARCH ON OCR TEXT
    clean_words = [w for w in re.findall(r'\w+', q_lower) if len(w) > 3 and w not in ["what", "when", "where", "which", "this", "that", "have", "from", "with"]]
    if clean_words and ocr_text:
        sentences = [s.strip() for s in re.split(r'[\r\n\.]+', ocr_text) if len(s.strip()) > 15]
        best_sentence = None
        best_matches = 0
        for sent in sentences:
            matches = sum(1 for w in clean_words if w in sent.lower())
            if matches > best_matches:
                best_matches = matches
                best_sentence = sent

        if best_sentence and best_matches > 0:
            if is_ta:
                return {"answer": f"ஆவணத்தில் கண்டறியப்பட்ட விவரம்: {best_sentence}", "lang": "ta", "intent": "keyword_search"}
            return {"answer": f"According to the document: {best_sentence}", "lang": "en", "intent": "keyword_search"}

    # Default fallback answer
    if is_ta:
        return {
            "answer": f"இந்த ஆவணம் {doc_type_ta} என வகைப்படுத்தப்பட்டுள்ளது. இதன் சுருக்கம், அடுத்த நடவடிக்கைகள், அல்லது தேதி பற்றி என்னிடம் கேட்கலாம்.",
            "lang": "ta",
            "intent": "fallback"
        }
    return {
        "answer": f"This document is classified as a {doc_type}. You can ask me about its summary, next action steps, dates, parties, or specific content.",
        "lang": "en",
        "intent": "fallback"
    }


# ============================================================
# GOVERNMENT SCHEME QUESTION ANSWERING ENGINE
# ============================================================

def answer_scheme_question(question, profile=None, lang="en"):
    """
    Analyzes questions about government schemes and returns helpful answers
    and recommendations using the GovEase hybrid schemes recommendation engine.
    """
    from scheme_recommendation import recommend_schemes

    q_text = (question or "").strip()
    is_ta = (lang == "ta") or is_tamil_text(question)

    if not q_text:
        if is_ta:
            return {"answer": "தயவுசெய்து நீங்கள் அறிய விரும்பும் திட்டத்தை பற்றி கூறவும் அல்லது தட்டச்சு செய்யவும்.", "lang": "ta"}
        return {"answer": "Please speak or type what government support or scheme you are looking for.", "lang": "en"}

    # Map query to category if identifiable
    category_key = "all"
    q_low = q_text.lower()
    if any(k in q_low for k in ["farm", "crop", "tractor", "agriculture", "fertilizer"]) or any(k in q_text for k in ["விவசாய", "பயிர்", "உழவர்"]):
        category_key = "agriculture"
    elif any(k in q_low for k in ["student", "school", "college", "scholarship", "education"]) or any(k in q_text for k in ["கல்வி", "மாணவ", "பள்ளி"]):
        category_key = "education"
    elif any(k in q_low for k in ["woman", "women", "girl", "mother", "maternity", "shg"]) or any(k in q_text for k in ["மகளிர்", "பெண்", "தாய்"]):
        category_key = "women"
    elif any(k in q_low for k in ["business", "loan", "startup", "msme", "enterprise"]) or any(k in q_text for k in ["தொழில்", "வணிக"]):
        category_key = "business"
    elif any(k in q_low for k in ["pension", "senior", "elderly", "old age", "widow"]) or any(k in q_text for k in ["ஓய்வூதிய", "முதியோர்"]):
        category_key = "pension"

    try:
        schemes = recommend_schemes(category_key, free_query=q_text, limit=3)
    except Exception as e:
        logger.error("Scheme matching error: %s", str(e))
        schemes = []

    if not schemes:
        if is_ta:
            return {
                "answer": "மன்னிக்கவும், உங்கள் கேள்விக்குரிய நேரடித் திட்டங்கள் எதுவும் பொருந்தவில்லை. பிற முக்கிய சொற்களைக் கொண்டு முயற்சிக்கவும் (எ.கா. விவசாயம், கல்வி உதவித்தொகை, மகளிர் நலன்).",
                "lang": "ta",
                "schemes": []
            }
        return {
            "answer": "I could not find exact schemes matching your query. Please try asking about areas like agriculture, education scholarships, pensions, or women welfare.",
            "lang": "en",
            "schemes": []
        }

    top_scheme = schemes[0]
    scheme_name = top_scheme.get("name", "Government Scheme")
    description = top_scheme.get("description", "")
    match_score = top_scheme.get("match_score", 85)
    state = top_scheme.get("state") or "All India"

    if is_ta:
        ans = f"உங்களுக்கு மிகவும் பொருத்தமான திட்டம் '{scheme_name}' ({state}). பொருத்தம் {match_score}%. விவரம்: {description[:160]}..."
        if len(schemes) > 1:
            ans += f" மேலும் '{schemes[1].get('name')}' திட்டமும் பரிந்துரைக்கப்படுகிறது."
    else:
        ans = f"The top matching scheme for you is '{scheme_name}' ({state}) with a {match_score}% match. Summary: {description[:160]}..."
        if len(schemes) > 1:
            ans += f" Another relevant scheme is '{schemes[1].get('name')}'.\""

    return {
        "answer": ans,
        "lang": "ta" if is_ta else "en",
        "top_scheme": top_scheme,
        "schemes": schemes[:3]
    }


# ============================================================
# API FLASK ROUTES
# ============================================================

@voice_bp.route("/status", methods=["GET"])
def voice_status():
    """Returns status and configuration of the voice assistant."""
    active_doc = False
    if _CURRENT_RESULT_GETTER:
        active_doc = bool(_CURRENT_RESULT_GETTER())

    return jsonify({
        "status": "online",
        "supported_languages": [
            {"code": "en-IN", "name": "English (India)"},
            {"code": "en-US", "name": "English (US)"},
            {"code": "ta-IN", "name": "தமிழ் (Tamil)"}
        ],
        "has_active_document": active_doc,
        "speech_synthesis": True,
        "speech_recognition": True
    })


@voice_bp.route("/ask", methods=["POST"])
def voice_ask():
    """
    Processes spoken or typed questions about document meaning or government schemes.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    context_type = data.get("context_type", "document").lower()
    lang = data.get("lang", "en").lower()
    profile = data.get("profile")

    if not question:
        return jsonify({
            "success": False,
            "error": "No question received. Please speak or type your question.",
            "answer": "Please ask a question about your document or government schemes."
        }), 400

    if context_type == "scheme":
        result = answer_scheme_question(question, profile=profile, lang=lang)
    else:
        doc_result = None
        if _CURRENT_RESULT_GETTER:
            doc_result = _CURRENT_RESULT_GETTER()

        # If question is explicitly about schemes even while on document page
        if any(w in question.lower() for w in ["scheme", "yojana", "subsidy", "scholarship", "pension"]) or \
           any(w in question for w in ["திட்டம்", "உதவித்தொகை", "மானியம்", "யோஜனா"]):
            result = answer_scheme_question(question, profile=profile, lang=lang)
            result["source"] = "scheme"
        else:
            result = answer_document_question(question, doc_result, lang=lang)
            result["source"] = "document"

    result["success"] = True
    result["question"] = question
    return jsonify(result)
