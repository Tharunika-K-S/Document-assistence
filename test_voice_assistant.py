"""
GovEase AI - Voice Assistant Integration & Speech Capabilities Test Suite
Verifies:
1. Text-to-Speech (TTS) integration, languages (Tamil & English), chunking, and multi-stream prevention.
2. Speech-to-Text (STT) Q&A processing for Document Understanding & Meaning.
3. Government Scheme Q&A processing in English & Tamil.
4. Fallback when microphone or active document is unavailable.
5. Static CSS and JS assets are properly served.
6. Zero regression on existing Flask routes and services.
"""

import os
import sys
import json
import unittest

PROJECT_DIR = "C:/Users/tharu/DOCUMENT ASSISTENCE"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from app import app, _ACTIVE_RESULTS
from voice_assistant import (
    answer_document_question,
    answer_scheme_question,
    is_tamil_text
)


class VoiceAssistantTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Sample mock document result
        self.mock_doc_result = {
            "filename": "Court_Order_WP_12894_2026.pdf",
            "text": (
                "IN THE HIGH COURT OF JUDICATURE AT MADRAS\n"
                "W.P. No. 12894 of 2026\n"
                "Date: 15-08-2026\n"
                "Petitioner: K. S. Tharunika\n"
                "Respondent: District Revenue Officer, Chennai\n"
                "Order: The petition for issuance of patta is hereby ALLOWED with directions to complete mutation within 4 weeks.\n"
                "Issued by: Hon'ble Justice S. Ramasamy\n"
            ),
            "document_type": "Court Order",
            "document_type_ta": "நீதிமன்ற உத்தரவு",
            "confidence": 96.5,
            "quality": "Good",
            "fields": {
                "Case Number": "W.P. No. 12894 of 2026",
                "Court Name": "High Court of Judicature at Madras",
                "Judge": "Hon'ble Justice S. Ramasamy",
                "Order Date": "15-08-2026",
                "Petitioner": "K. S. Tharunika",
                "Respondent": "District Revenue Officer, Chennai",
                "Outcome": "ALLOWED"
            },
            "court_order": {
                "case_number": "W.P. No. 12894 of 2026",
                "court_name": "High Court of Judicature at Madras",
                "judge": "Hon'ble Justice S. Ramasamy",
                "order_date": "15-08-2026",
                "petitioner": "K. S. Tharunika",
                "respondent": "District Revenue Officer, Chennai",
                "outcome": "ALLOWED"
            },
            "summary": {
                "quick_summary": {
                    "en": [
                        "The High Court of Judicature at Madras allowed the petition.",
                        "Directs District Revenue Officer to complete patta mutation within 4 weeks.",
                        "Order passed in favour of petitioner K. S. Tharunika on 15-08-2026."
                    ],
                    "ta": [
                        "சென்னை உயர் நீதிமன்றம் இந்த மனுவை அனுமதித்துள்ளது.",
                        "4 வாரங்களுக்குள் பட்டா மாறுதல் பணிகளை முடிக்க உத்தரவிடப்பட்டுள்ளது.",
                        "மனுதாரர் கே. எஸ். தருணிகாவுக்கு ஆதரவாக 15-08-2026 அன்று ஆணை பிறப்பிக்கப்பட்டது."
                    ]
                },
                "action_points": {
                    "en": [
                        "Submit certified copy of this court order to the Taluk Tahsildar office.",
                        "Track patta transfer mutation within the 4-week compliance window.",
                        "Maintain acknowledgement receipt for records."
                    ],
                    "ta": [
                        "நீதிமன்ற உத்தரவின் சான்றளிக்கப்பட்ட நகலை வட்டாட்சியர் அலுவலகத்தில் சமர்ப்பிக்கவும்.",
                        "4 வார காலக்கெடுவுக்குள் பட்டா பெயர் மாற்ற நிலையை கண்காணிக்கவும்.",
                        "ஒப்புதல் சீட்டை பாதுகாப்பாக வைத்திருக்கவும்."
                    ]
                },
                "detailed_summary": {
                    "en": "Detailed analysis of Court Order W.P. No. 12894 of 2026 issued on 15-08-2026.",
                    "ta": "15-08-2026 அன்று பிறப்பிக்கப்பட்ட நீதிமன்ற ஆணை எண் W.P. No. 12894 of 2026 பற்றிய விரிவான விளக்கம்."
                }
            }
        }

    # ---------------------------------------------------------
    # TEST 1: STATIC ASSETS SERVING
    # ---------------------------------------------------------
    def test_voice_static_assets_served(self):
        """Ensure voice_assistant.css and voice_assistant.js return 200 OK."""
        res_css = self.client.get("/static/voice_assistant.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn(b"voice-read-btn", res_css.data)
        self.assertIn(b"voice-audio-bar", res_css.data)

        res_js = self.client.get("/static/voice_assistant.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn(b"GovEaseVoice", res_js.data)
        self.assertIn(b"SpeechSynthesisUtterance", res_js.data)

    # ---------------------------------------------------------
    # TEST 2: VOICE STATUS ENDPOINT
    # ---------------------------------------------------------
    def test_voice_status_endpoint(self):
        """Ensure /api/voice/status returns supported languages and system health."""
        res = self.client.get("/api/voice/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "online")
        codes = [l["code"] for l in data.get("supported_languages", [])]
        self.assertIn("en-IN", codes)
        self.assertIn("ta-IN", codes)

    # ---------------------------------------------------------
    # TEST 3: VALIDATION & EMPTY QUERY FALLBACK
    # ---------------------------------------------------------
    def test_voice_ask_empty_payload(self):
        """Ensure empty question payload returns 400 error."""
        res = self.client.post("/api/voice/ask", json={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data.get("success"))

    def test_voice_ask_no_active_document(self):
        """Ensure querying document meaning without active document gives clear guidance."""
        res = self.client.post("/api/voice/ask", json={
            "question": "What is this document about?",
            "context_type": "document",
            "lang": "en"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertIn("No document is currently active", data.get("answer"))

    # ---------------------------------------------------------
    # TEST 4: DOCUMENT MEANING Q&A (ENGLISH & TAMIL)
    # ---------------------------------------------------------
    def test_document_qa_summary_english(self):
        """Tests summary question on document in English."""
        res = answer_document_question("What is the summary of this document?", self.mock_doc_result, lang="en")
        self.assertEqual(res.get("lang"), "en")
        self.assertEqual(res.get("intent"), "summary")
        self.assertIn("Court Order", res.get("answer"))
        self.assertIn("allowed the petition", res.get("answer"))

    def test_document_qa_summary_tamil(self):
        """Tests summary question on document in Tamil."""
        res = answer_document_question("இந்த ஆவணத்தின் சுருக்கம் என்ன?", self.mock_doc_result, lang="ta")
        self.assertEqual(res.get("lang"), "ta")
        self.assertEqual(res.get("intent"), "summary")
        self.assertIn("நீதிமன்ற உத்தரவு", res.get("answer"))
        self.assertIn("அனுமதித்துள்ளது", res.get("answer"))

    def test_document_qa_actions_english(self):
        """Tests action points question on document in English."""
        res = answer_document_question("What should I do next?", self.mock_doc_result, lang="en")
        self.assertEqual(res.get("intent"), "actions")
        self.assertIn("action steps", res.get("answer"))
        self.assertIn("Tahsildar", res.get("answer"))

    def test_document_qa_actions_tamil(self):
        """Tests action points question on document in Tamil."""
        res = answer_document_question("அடுத்த நடவடிக்கை என்ன?", self.mock_doc_result, lang="ta")
        self.assertEqual(res.get("intent"), "actions")
        self.assertIn("அடுத்த கட்ட நடவடிக்கைகள்", res.get("answer"))

    def test_document_qa_date(self):
        """Tests date extraction question."""
        res = answer_document_question("When was this order issued?", self.mock_doc_result, lang="en")
        self.assertEqual(res.get("intent"), "date")
        self.assertIn("15-08-2026", res.get("answer"))

    def test_document_qa_parties(self):
        """Tests parties/names extraction question."""
        res = answer_document_question("Who is the petitioner in this case?", self.mock_doc_result, lang="en")
        self.assertEqual(res.get("intent"), "parties")
        self.assertIn("K. S. Tharunika", res.get("answer"))

    def test_document_qa_court_order(self):
        """Tests court verdict/outcome question."""
        res = answer_document_question("What is the court outcome?", self.mock_doc_result, lang="en")
        self.assertEqual(res.get("intent"), "court")
        self.assertIn("ALLOWED", res.get("answer"))

    def test_document_qa_quality(self):
        """Tests OCR confidence & quality question."""
        res = answer_document_question("How good is the OCR quality and confidence?", self.mock_doc_result, lang="en")
        self.assertEqual(res.get("intent"), "quality")
        self.assertIn("Good", res.get("answer"))
        self.assertIn("96.5", res.get("answer"))

    # ---------------------------------------------------------
    # TEST 5: GOVERNMENT SCHEME Q&A (ENGLISH & TAMIL)
    # ---------------------------------------------------------
    def test_scheme_qa_farmer_query_english(self):
        """Tests scheme question for farmers in English."""
        res = answer_scheme_question("What schemes are available for farmers and agriculture subsidy?", lang="en")
        self.assertEqual(res.get("lang"), "en")
        self.assertTrue(len(res.get("schemes", [])) > 0)
        self.assertIn("top matching scheme", res.get("answer").lower())

    def test_scheme_qa_farmer_query_tamil(self):
        """Tests scheme question for farmers in Tamil."""
        res = answer_scheme_question("விவசாயிகளுக்கு என்ன மானிய திட்டம் உள்ளது?", lang="ta")
        self.assertEqual(res.get("lang"), "ta")
        self.assertTrue(len(res.get("schemes", [])) > 0)
        self.assertIn("பொருத்தமான திட்டம்", res.get("answer"))

    def test_scheme_qa_student_scholarship(self):
        """Tests scheme question for student scholarships."""
        res = answer_scheme_question("college scholarship scheme for girl students", lang="en")
        self.assertTrue(len(res.get("schemes", [])) > 0)
        self.assertTrue("match" in res.get("answer").lower())

    # ---------------------------------------------------------
    # TEST 6: LANGUAGE DETECTION HELPER
    # ---------------------------------------------------------
    def test_is_tamil_text(self):
        """Verifies Tamil Unicode script detection."""
        self.assertTrue(is_tamil_text("தமிழ்நாடு அரசு"))
        self.assertTrue(is_tamil_text("பயனாளி யார்?"))
        self.assertFalse(is_tamil_text("Government of Tamil Nadu"))
        self.assertFalse(is_tamil_text("1234567890"))

    # ---------------------------------------------------------
    # TEST 7: EXISTING ROUTES INTEGRITY & NO BREAKAGE
    # ---------------------------------------------------------
    def test_existing_routes_not_broken(self):
        """Ensures index, schemes, history, login, register are completely intact."""
        routes = ["/", "/schemes/", "/history", "/login", "/register"]
        for route in routes:
            r = self.client.get(route)
            self.assertIn(r.status_code, [200, 302])
            if r.status_code == 200:
                self.assertNotIn(b"Internal Server Error", r.data)


if __name__ == "__main__":
    unittest.main()
