"""
GovEase AI - Required Documents Checklist Test Suite
Tests:
1. Parsing of raw document requirements into structured checklist items.
2. Recognition of common document types (Aadhaar, income, caste, residence, bank passbook, educational certificates, photos).
3. Handling of schemes with missing/null/empty requirements safely without inventing requirements.
4. "Requirements not available—please verify with the official scheme source." fallback.
5. Official source URL presence for verified linking.
6. API endpoint /schemes/api/checklist/<index> response format.
7. Presence of checklist section, buttons, and disclaimers in templates/schemes.html.
8. Preservation of all existing features.
"""

import os
import sys
import unittest

PROJECT_DIR = "C:/Users/tharu/DOCUMENT ASSISTENCE"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from app import app
from scheme_recommendation import (
    load_schemes,
    parse_scheme_documents_checklist,
    recommend_schemes
)


class RequiredDocumentsChecklistTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.schemes = load_schemes()

    # ---------------------------------------------------------
    # TEST 1: COMMON DOCUMENT TYPE PARSING & ENRICHMENT
    # ---------------------------------------------------------
    def test_common_documents_enrichment(self):
        """Verify that common documents (Aadhaar, income, caste, residence, bank passbook, marksheet, photo) are parsed and assigned clear explanations."""
        sample_raw = (
            "Aadhaar Card of the applicant; "
            "Income Certificate issued by Tahsildar; "
            "Caste Certificate (SC/ST/OBC); "
            "Residence / Domicile Certificate; "
            "Bank Account Passbook with IFSC; "
            "10th / 12th Marksheet; "
            "Recent Passport-size Photograph; "
            "Land Patta / Chitta document"
        )
        res = parse_scheme_documents_checklist(sample_raw)
        self.assertTrue(res["has_documents"])
        self.assertIsNone(res["notice"])
        self.assertEqual(len(res["items"]), 8)

        names = [item["name"] for item in res["items"]]
        self.assertTrue(any("Aadhaar" in n for n in names))
        self.assertTrue(any("Income" in n for n in names))
        self.assertTrue(any("Caste" in n for n in names))
        self.assertTrue(any("Residence" in n or "Domicile" in n for n in names))
        self.assertTrue(any("Bank" in n for n in names))
        self.assertTrue(any("Marksheet" in n for n in names))
        self.assertTrue(any("Passport" in n or "Photo" in n for n in names))
        self.assertTrue(any("Patta" in n or "Land" in n for n in names))

        # Check explanations are informative
        for item in res["items"]:
            self.assertTrue(len(item["explanation"]) > 15)

    # ---------------------------------------------------------
    # TEST 2: SAFE HANDLING OF MISSING / NULL / EMPTY REQUIREMENTS
    # ---------------------------------------------------------
    def test_missing_requirements_handling(self):
        """Verify that schemes without requirements return has_documents=False and the exact mandated notice."""
        expected_notice = "Requirements not available—please verify with the official scheme source."
        
        for empty_val in [None, "", "   ", "nan", "NaN", "null", "none", "[]"]:
            res = parse_scheme_documents_checklist(empty_val)
            self.assertFalse(res["has_documents"], f"Failed for {empty_val}")
            self.assertEqual(res["items"], [])
            self.assertIn(expected_notice, res["notice"])

    # ---------------------------------------------------------
    # TEST 3: REAL DATASET SCHEMES WITH AND WITHOUT REQUIREMENTS
    # ---------------------------------------------------------
    def test_real_schemes_parsing(self):
        """Verify real schemes from the dataset are parsed properly."""
        with_docs = [s for s in self.schemes if s.get("documents_required") and str(s.get("documents_required")).strip() not in ["", "nan", "null", "none"]]
        without_docs = [s for s in self.schemes if not s.get("documents_required") or str(s.get("documents_required")).strip() in ["", "nan", "null", "none"]]

        self.assertTrue(len(with_docs) > 4000, "Should have over 4000 schemes with document requirements")
        self.assertTrue(len(without_docs) > 100, "Should have schemes without document requirements")

        # Test scheme WITH documents
        sample_with = with_docs[0]
        res_with = parse_scheme_documents_checklist(sample_with.get("documents_required"))
        self.assertTrue(res_with["has_documents"])
        self.assertTrue(len(res_with["items"]) > 0)
        self.assertTrue("source_url" in sample_with)

        # Test scheme WITHOUT documents
        sample_without = without_docs[0]
        res_without = parse_scheme_documents_checklist(sample_without.get("documents_required"))
        self.assertFalse(res_without["has_documents"])
        self.assertEqual(len(res_without["items"]), 0)
        self.assertIn("Requirements not available—please verify with the official scheme source.", res_without["notice"])

    # ---------------------------------------------------------
    # TEST 4: RECOMMEND SCHEMES PAYLOAD INTEGRATION
    # ---------------------------------------------------------
    def test_recommend_schemes_includes_checklist_and_url(self):
        """Verify recommend_schemes returns documents_checklist and source_url."""
        results = recommend_schemes("agriculture", free_query="tractor", limit=3)
        self.assertTrue(len(results) > 0)
        for s in results:
            self.assertIn("documents_checklist", s)
            chk = s["documents_checklist"]
            self.assertIn("has_documents", chk)
            self.assertIn("items", chk)
            self.assertIn("source_url", s)
            self.assertTrue(len(s["source_url"]) > 0)

    # ---------------------------------------------------------
    # TEST 5: API ENDPOINT /schemes/api/checklist/<index>
    # ---------------------------------------------------------
    def test_checklist_api_endpoint(self):
        """Verify GET /schemes/api/checklist/<index> endpoint."""
        r = self.client.get("/schemes/api/checklist/0")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data["success"])
        self.assertIn("scheme_name", data)
        self.assertIn("source_url", data)
        self.assertIn("checklist", data)
        self.assertIn("has_documents", data["checklist"])

        # Test out of range index
        r_404 = self.client.get("/schemes/api/checklist/999999")
        self.assertEqual(r_404.status_code, 404)

    # ---------------------------------------------------------
    # TEST 6: TEMPLATE UI ELEMENTS & DISCLAIMER
    # ---------------------------------------------------------
    def test_template_ui_elements(self):
        """Verify required HTML elements, disclaimers, and styles exist in schemes.html."""
        res = self.client.get("/schemes/")
        self.assertEqual(res.status_code, 200)
        content = res.data.decode("utf-8")

        self.assertIn("checklist-section", content)
        self.assertIn("checklistSection", content)
        self.assertIn("openDocumentChecklist", content)
        self.assertIn("btn-checklist", content)
        self.assertIn("btn-official-source", content)
        self.assertIn("checklist-status-btn", content)
        self.assertIn("Required Documents Checklist", content)
        self.assertIn("Requirements not available—please verify with the official scheme source", content)
        self.assertIn("Important Preparation Aid", content)
        self.assertIn("does not constitute official confirmation of eligibility", content)


if __name__ == "__main__":
    unittest.main()
