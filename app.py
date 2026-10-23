"""
GovEase AI — Bilingual Government Document Understanding and Scheme Recommendation Assistant
Main Flask Application
"""

import os
import io
import time
import json
import uuid
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_from_directory, send_file, jsonify, make_response
)
from werkzeug.utils import secure_filename

# Modular subsystem imports
from auth_system import (
    auth_bp, init_auth_db, login_required, current_user,
    save_document_history, get_user_history, get_history_entry, delete_history_entry
)
from scheme_recommendation import scheme_bp
from readability_analyzer import analyze_readability
from ocr_pipeline import process_image_file, process_pdf_file, OCR_LANGUAGE, HAS_TAMIL
from document_classifier import classify_document, extract_general_fields, build_fields_i18n, DOCUMENT_TYPES
from court_processor import simplify_court_order
from summarizer import summarize_document
from pdf_exporter import generate_pdf_report

# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "govease-ai-secret-key-production-college-demo-2026"
)

# Register Blueprints
app.register_blueprint(scheme_bp)
app.register_blueprint(auth_bp)

# Initialize Database
init_auth_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

# Server-side cache for current active document results (prevents cookie overflow)
_ACTIVE_RESULTS = {}

def get_current_result():
    token = session.get("doc_token")
    if token and token in _ACTIVE_RESULTS:
        return _ACTIVE_RESULTS[token]
    user = current_user()
    last_id = session.get("last_history_id")
    if user and last_id:
        entry = get_history_entry(last_id, user["id"])
        if entry:
            return entry.get("result", {})
    return None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# CONTEXT PROCESSORS & ERROR HANDLERS
# ============================================================

@app.context_processor
def inject_global_data():
    return {
        "current_user": current_user(),
        "has_tamil_ocr": HAS_TAMIL,
        "ocr_language": OCR_LANGUAGE
    }

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("The uploaded file exceeds the 16MB maximum limit. Please upload a smaller file.", "error")
    return redirect(url_for("index"))

@app.errorhandler(404)
def page_not_found(error):
    return render_template("index.html", error="The requested page could not be found."), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template("index.html", error="An internal system error occurred. Please try again with a clearer document."), 500


# ============================================================
# MAIN DOCUMENT ANALYSIS ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No document file was selected. Please choose a file to upload."
            return render_template("index.html", error=error, result=None)

        file = request.files["file"]

        if not file or file.filename.strip() == "":
            error = "Please select a valid document file (JPG, PNG, or PDF)."
            return render_template("index.html", error=error, result=None)

        if not allowed_file(file.filename):
            error = "Unsupported file type. Please upload a scanned document or image in JPG, JPEG, PNG, or PDF format."
            return render_template("index.html", error=error, result=None)

        raw_filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{raw_filename}"
        upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        try:
            file.save(upload_path)
            extension = raw_filename.rsplit(".", 1)[1].lower()

            # 1. OCR Extraction (Images & PDFs)
            if extension == "pdf":
                ocr_data = process_pdf_file(upload_path, raw_filename, PROCESSED_FOLDER)
            else:
                ocr_data = process_image_file(upload_path, raw_filename, PROCESSED_FOLDER)

            extracted_text = ocr_data["text"]

            if not extracted_text or len(extracted_text.strip()) < 10:
                error = "Could not extract readable text from the document. Please ensure the document is clear, well-lit, and not overly blurry."
                return render_template("index.html", error=error, result=None)

            # 2. Document Classification
            doc_type, class_conf, matched_kws = classify_document(extracted_text)
            doc_type_ta = DOCUMENT_TYPES.get(doc_type, "பொது அரசு ஆவணம்")

            # 3. Field Extraction
            fields = extract_general_fields(extracted_text, doc_type)

            # 4. Court Order Handling
            court_order = None
            if doc_type == "Court Order":
                court_order = simplify_court_order(extracted_text)
                fields["Case Number"] = court_order["case_number"]
                fields["Court Name"] = court_order["court_name"]
                fields["Judge"] = court_order["judge"]
                fields["Order Date"] = court_order["order_date"]
                fields["Petitioner"] = court_order["petitioner"]
                fields["Respondent"] = court_order["respondent"]
                fields["Outcome"] = court_order["outcome"]

            fields_i18n = build_fields_i18n(fields)

            # 5. Readability Analysis
            readability = analyze_readability(extracted_text)

            # 6. AI Summarization (Quick Summary, Detailed Summary, Action Points in EN & TA)
            summary = summarize_document(extracted_text, doc_type, fields)

            result = {
                "filename": raw_filename,
                "text": extracted_text,
                "confidence": ocr_data["confidence"],
                "quality": ocr_data["quality"],
                "warning": ocr_data.get("warning"),
                "best_version": ocr_data["best_version"],
                "best_psm": ocr_data.get("best_psm"),
                "document_detected": ocr_data.get("document_detected", True),
                "processed_filename": ocr_data.get("processed_filename"),
                "document_type": doc_type,
                "document_type_ta": doc_type_ta,
                "classification_confidence": class_conf,
                "classification_keywords": matched_kws,
                "fields": fields,
                "fields_i18n": fields_i18n,
                "court_order": court_order,
                "readability": readability,
                "summary": summary
            }

            # 7. Auto-save to Document History for logged in users
            user = current_user()
            if user:
                history_id = save_document_history(
                    user_id=user["id"],
                    filename=raw_filename,
                    document_type=doc_type,
                    document_type_ta=doc_type_ta,
                    confidence=ocr_data["confidence"],
                    processed_filename=ocr_data.get("processed_filename", ""),
                    result_dict=result
                )
                result["history_id"] = history_id
                session["last_history_id"] = history_id
                flash("Document processed and saved to your history successfully.", "info")

            # Store token in server cache & session cookie
            token = str(uuid.uuid4())
            _ACTIVE_RESULTS[token] = result
            session["doc_token"] = token

        except Exception as exc:
            print("PROCESSING ERROR:", exc)
            error = f"Document processing failed: {str(exc)}. Please check the file and try again."

    return render_template("index.html", error=error, result=result)


# ============================================================
# DOCUMENT HISTORY & VIEW ROUTES
# ============================================================

@app.route("/history")
@login_required
def history():
    user = current_user()
    user_history = get_user_history(user["id"])
    return render_template("history.html", history_items=user_history)


@app.route("/document/<int:history_id>")
@login_required
def view_document(history_id):
    user = current_user()
    entry = get_history_entry(history_id, user["id"])
    if not entry:
        flash("Document history record not found or access denied.", "error")
        return redirect(url_for("history"))

    result = entry.get("result", {})
    result["history_id"] = history_id
    session["last_history_id"] = history_id
    token = str(uuid.uuid4())
    _ACTIVE_RESULTS[token] = result
    session["doc_token"] = token

    return render_template("index.html", result=result, from_history=True)


@app.route("/history/delete/<int:history_id>", methods=["POST"])
@login_required
def delete_history(history_id):
    user = current_user()
    deleted = delete_history_entry(history_id, user["id"])
    if deleted:
        flash("Document history entry deleted successfully.", "info")
    else:
        flash("Unable to delete document history entry.", "error")
    return redirect(url_for("history"))


# ============================================================
# SUMMARY EXPORT ROUTES (PDF & TEXT)
# ============================================================

@app.route("/export_pdf/<int:history_id>")
@login_required
def export_pdf_history(history_id):
    user = current_user()
    entry = get_history_entry(history_id, user["id"])
    if not entry:
        flash("Document not found.", "error")
        return redirect(url_for("history"))

    result = entry.get("result", {})
    pdf_bytes = generate_pdf_report(result, entry.get("filename", "Document"))

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"GovEase_Summary_{history_id}.pdf"
    )


@app.route("/export_pdf_direct")
def export_pdf_direct():
    result = get_current_result()
    if not result:
        flash("No recent document summary found to export.", "error")
        return redirect(url_for("index"))

    pdf_bytes = generate_pdf_report(result, result.get("filename", "Document"))
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"GovEase_Summary_{int(time.time())}.pdf"
    )


@app.route("/export_text")
def export_text():
    result = get_current_result()
    if not result:
        flash("No active document summary found to export.", "error")
        return redirect(url_for("index"))

    summary = result.get("summary", {})
    quick_en = summary.get("quick_summary", {}).get("en", [])
    quick_ta = summary.get("quick_summary", {}).get("ta", [])
    act_en = summary.get("action_points", {}).get("en", [])
    act_ta = summary.get("action_points", {}).get("ta", [])
    det_en = summary.get("detailed_summary", {}).get("en", "")
    det_ta = summary.get("detailed_summary", {}).get("ta", "")

    content = f"""============================================================
GOVEASE AI — BILINGUAL GOVERNMENT DOCUMENT SUMMARY REPORT
============================================================
File: {result.get('filename')}
Document Type: {result.get('document_type')} ({result.get('document_type_ta')})
OCR Quality: {result.get('quality')} (Confidence: {result.get('confidence')}%)
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

------------------------------------------------------------
1. QUICK SUMMARY (ENGLISH):
------------------------------------------------------------
""" + "\n".join(f"• {b}" for b in quick_en) + f"""

------------------------------------------------------------
1. சுருக்கக் குறிப்புகள் (தமிழ்):
------------------------------------------------------------
""" + "\n".join(f"• {b}" for b in quick_ta) + f"""

------------------------------------------------------------
2. NEXT ACTION POINTS / அடுத்த கட்ட நடவடிக்கைகள்:
------------------------------------------------------------
(English):
""" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(act_en)) + f"""

(தமிழ்):
""" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(act_ta)) + f"""

------------------------------------------------------------
3. DETAILED EXPLANATION:
------------------------------------------------------------
{det_en}

------------------------------------------------------------
4. EXTRACTED INFORMATION:
------------------------------------------------------------
""" + "\n".join(f"• {f.get('label_en')} / {f.get('label_ta')}: {f.get('value')}" for f in result.get('fields_i18n', [])) + f"""

============================================================
Disclaimer: Informational guidance only. Verify with authorities.
============================================================
"""

    response = make_response(content)
    response.headers["Content-Disposition"] = f"attachment; filename=GovEase_Summary_{int(time.time())}.txt"
    response.headers["Content-Type"] = "text/plain; charset=utf-8"
    return response


# ============================================================
# SERVE PROCESSED PREVIEW FILES
# ============================================================

@app.route("/processed/<filename>")
def processed_file(filename):
    safe_name = secure_filename(filename)
    return send_from_directory(PROCESSED_FOLDER, safe_name)


# ============================================================
# APPLICATION ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GovEase AI — Bilingual Government Document Understanding")
    print("OCR Language:", OCR_LANGUAGE, "| Tamil Support:", "YES" if HAS_TAMIL else "NO (eng fallback)")
    print("Web URL: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )