"""
GovEase AI - Bilingual PDF Summary Report Generator
Generates clean, professionally formatted PDF reports containing
document overview, Quick Summary, Detailed Summary, Action Points,
and Extracted Fields in both English and Tamil.
"""

import io
import os
import pymupdf

NIRMALA_FONT_PATH = r"C:\Windows\Fonts\Nirmala.ttc"

def get_font_file():
    if os.path.exists(NIRMALA_FONT_PATH):
        return NIRMALA_FONT_PATH
    return None

def generate_pdf_report(result, original_filename="Document"):
    """
    Generates a bilingual PDF report from document analysis results.
    Returns bytes of the generated PDF.
    """
    doc = pymupdf.open()
    font_path = get_font_file()
    fontname = "default_font"
    
    page = doc.new_page(width=595, height=842) # A4
    if font_path:
        try:
            page.insert_font(fontname=fontname, fontfile=font_path)
        except Exception:
            fontname = "helv"
    else:
        fontname = "helv"

    y = 50
    margin_left = 50
    content_width = 495

    # Top Banner / Header
    page.draw_rect(pymupdf.Rect(40, 35, 555, 80), color=(0.06, 0.12, 0.27), fill=(0.06, 0.12, 0.27))
    page.insert_text(
        (margin_left, 56),
        "GovEase AI — Government Document Summary Report",
        fontname=fontname,
        fontsize=14,
        color=(1, 1, 1)
    )
    page.insert_text(
        (margin_left, 72),
        "Bilingual Document Analysis & Scheme Guidance Portal",
        fontname=fontname,
        fontsize=9,
        color=(0.8, 0.88, 1)
    )

    y = 105

    # Metadata Block
    doc_type = result.get("document_type", "General Government Document")
    doc_type_ta = result.get("document_type_ta", "பொது அரசு ஆவணம்")
    conf = result.get("confidence", 0)
    quality = result.get("quality", "Good")

    page.insert_text((margin_left, y), f"File: {original_filename}", fontname=fontname, fontsize=10, color=(0.1, 0.1, 0.1))
    y += 16
    page.insert_text((margin_left, y), f"Document Type: {doc_type} ({doc_type_ta})", fontname=fontname, fontsize=10, color=(0.15, 0.35, 0.8))
    y += 16
    page.insert_text((margin_left, y), f"OCR Confidence: {conf}% | Quality: {quality}", fontname=fontname, fontsize=9, color=(0.3, 0.3, 0.3))
    y += 24

    # Separator Line
    page.draw_line(pymupdf.Point(margin_left, y), pymupdf.Point(margin_left + content_width, y), color=(0.85, 0.88, 0.92))
    y += 20

    summaries = result.get("summary", {})
    quick_sum = summaries.get("quick_summary", {})
    detailed_sum = summaries.get("detailed_summary", {})
    action_pts = summaries.get("action_points", {})

    # SECTION 1: QUICK SUMMARY (ENGLISH & TAMIL)
    page.insert_text((margin_left, y), "1. Quick Summary / சுருக்கக் குறிப்புகள்:", fontname=fontname, fontsize=12, color=(0.06, 0.12, 0.27))
    y += 18

    quick_en_list = quick_sum.get("en", [])
    for bullet in quick_en_list[:4]:
        page.insert_text((margin_left + 10, y), f"• {bullet}", fontname=fontname, fontsize=9, color=(0.15, 0.15, 0.15))
        y += 15

    quick_ta_list = quick_sum.get("ta", [])
    if quick_ta_list and fontname != "helv":
        y += 6
        for bullet in quick_ta_list[:3]:
            page.insert_text((margin_left + 10, y), f"• {bullet}", fontname=fontname, fontsize=9, color=(0.2, 0.2, 0.2))
            y += 15

    y += 15
    page.draw_line(pymupdf.Point(margin_left, y), pymupdf.Point(margin_left + content_width, y), color=(0.85, 0.88, 0.92))
    y += 20

    # SECTION 2: ACTION POINTS
    page.insert_text((margin_left, y), "2. Next Action Steps / அடுத்த கட்ட நடவடிக்கைகள்:", fontname=fontname, fontsize=12, color=(0.06, 0.12, 0.27))
    y += 18

    act_en_list = action_pts.get("en", [])
    for idx, act in enumerate(act_en_list[:3], 1):
        page.insert_text((margin_left + 10, y), f"{idx}. {act}", fontname=fontname, fontsize=9, color=(0.15, 0.15, 0.15))
        y += 15

    act_ta_list = action_pts.get("ta", [])
    if act_ta_list and fontname != "helv":
        y += 6
        for idx, act in enumerate(act_ta_list[:2], 1):
            page.insert_text((margin_left + 10, y), f"{idx}. {act}", fontname=fontname, fontsize=9, color=(0.2, 0.2, 0.2))
            y += 15

    y += 15
    page.draw_line(pymupdf.Point(margin_left, y), pymupdf.Point(margin_left + content_width, y), color=(0.85, 0.88, 0.92))
    y += 20

    # SECTION 3: EXTRACTED INFORMATION
    page.insert_text((margin_left, y), "3. Extracted Information / பிரித்தெடுக்கப்பட்ட தகவல்கள்:", fontname=fontname, fontsize=12, color=(0.06, 0.12, 0.27))
    y += 18

    fields_i18n = result.get("fields_i18n", [])
    if not fields_i18n and "fields" in result:
        fields_i18n = [{"label_en": k, "label_ta": k, "value": v} for k, v in result["fields"].items()]

    for field in fields_i18n[:8]:
        lbl = field.get("label_en", "")
        val = field.get("value", "Not Detected")
        page.insert_text((margin_left + 10, y), f"• {lbl}: {val}", fontname=fontname, fontsize=9, color=(0.2, 0.2, 0.2))
        y += 15

    # Footer Disclaimer
    page.draw_line(pymupdf.Point(margin_left, 800), pymupdf.Point(margin_left + content_width, 800), color=(0.8, 0.8, 0.8))
    page.insert_text(
        (margin_left, 815),
        "Disclaimer: This summary is generated by GovEase AI for informational purposes. Verify all official eligibility with the respective department.",
        fontname=fontname,
        fontsize=7,
        color=(0.5, 0.5, 0.5)
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
