"""
GovEase AI - Comprehensive Upload Reliability and PDF Verification Test Suite
Tests all 13+ edge cases required by the prompt.
"""

import io
import os
import sys
import unittest
from PIL import Image, ImageDraw

# Set up environment path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from app import app, ALLOWED_EXTENSIONS, normalize_extension, allowed_file, validate_uploaded_file


def create_test_text_pdf(text=None, num_pages=1):
    """Generates an in-memory PDF with direct text."""
    doc = fitz.open()
    sample_text = text or (
        "GOVERNMENT OF TAMIL NADU\n"
        "Revenue & Disaster Management Department\n"
        "Patta Transfer Approval Order\n"
        "District: Chennai, Taluk: Guindy, Village: Adyar\n"
        "Applicant: Tharunika K. S.\n"
        "Patta Number: 45892, Survey Number: 104/2B\n"
        "Extent: 0.04.5 Hectares\n"
        "Order Date: 24-10-2026\n"
        "This is to certify that the patta transfer application has been approved."
    )
    for p in range(num_pages):
        page = doc.new_page()
        page.insert_text((50, 72), f"Page {p+1}\n" + sample_text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.getvalue()


def create_test_scanned_pdf():
    """Generates an in-memory scanned PDF (page containing an image of printed text)."""
    img = Image.new("RGB", (1200, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "GOVERNMENT OF TAMIL NADU",
        "REVENUE DEPARTMENT CERTIFICATE",
        "Certificate No: TN-REV-2026-987654",
        "This document certifies first graduate status.",
        "Applicant Name: K. S. Tharunika",
        "Date of Issue: 24-10-2026"
    ]
    y = 50
    for line in lines:
        draw.text((60, y), line, fill=(0, 0, 0))
        y += 45

    img_buf = io.BytesIO()
    img.save(img_buf, format="JPEG", quality=95)
    img_bytes = img_buf.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(50, 50, 545, 400), stream=img_bytes)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.getvalue()


def create_test_image(format="JPEG"):
    """Generates a test image with clear text."""
    img = Image.new("RGB", (1000, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "TAMIL NADU GOVERNMENT",
        "COMMUNITY CERTIFICATE",
        "Application Number: CC20260401",
        "Date of Order: 24-10-2026",
        "Name: Tharunika",
        "Community: Backward Class"
    ]
    y = 40
    for line in lines:
        draw.text((50, y), line, fill=(0, 0, 0))
        y += 40

    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


class UploadReliabilityTestCase(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    def test_01_valid_text_pdf_first_attempt(self):
        """1. A valid text-based PDF on the first attempt."""
        pdf_bytes = create_test_text_pdf()
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(pdf_bytes), "patta_order.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html)
        self.assertIn("resultsSection", html)
        self.assertIn("patta_order.pdf", html)

    def test_02_valid_scanned_pdf_first_attempt(self):
        """2. A valid scanned PDF on the first attempt."""
        pdf_bytes = create_test_scanned_pdf()
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(pdf_bytes), "scanned_doc.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html)
        self.assertIn("resultsSection", html)

    def test_03_multi_page_pdf(self):
        """3. A multi-page PDF (3 pages)."""
        pdf_bytes = create_test_text_pdf(num_pages=3)
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(pdf_bytes), "multipage_record.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html)
        self.assertIn("resultsSection", html)

    def test_04_uppercase_pdf_extension(self):
        """4. An uppercase .PDF filename."""
        pdf_bytes = create_test_text_pdf()
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(pdf_bytes), "REVENUE_ORDER.PDF", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html)
        self.assertIn("resultsSection", html)

    def test_05_valid_jpg_and_png(self):
        """5. A valid JPG and PNG."""
        jpg_bytes = create_test_image("JPEG")
        resp_jpg = self.client.post(
            "/",
            data={"file": (io.BytesIO(jpg_bytes), "doc_photo.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp_jpg.status_code, 200)
        html_jpg = resp_jpg.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html_jpg)
        self.assertIn("resultsSection", html_jpg)

        png_bytes = create_test_image("PNG")
        resp_png = self.client.post(
            "/",
            data={"file": (io.BytesIO(png_bytes), "certificate.png", "image/png")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp_png.status_code, 200)
        html_png = resp_png.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html_png)
        self.assertIn("resultsSection", html_png)

    def test_06_unsupported_file_type(self):
        """6. An unsupported file type."""
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(b"random text data"), "archive.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertIn("<strong>Error:</strong>", html)
        self.assertIn("Unsupported file format (.docx)", html)

    def test_07_corrupted_pdf(self):
        """7. A corrupted PDF."""
        corrupt_bytes = b"%PDF-1.4\nthis is truncated corrupt binary content without xref or trailer"
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(corrupt_bytes), "broken.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertIn("<strong>Error:</strong>", html)
        self.assertTrue("corrupted" in html.lower() or "failed" in html.lower())

    def test_08_empty_file(self):
        """8. An empty file (0 bytes)."""
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(b""), "empty.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertIn("<strong>Error:</strong>", html)
        self.assertIn("empty (0 bytes)", html)

    def test_09_extension_content_mismatch(self):
        """9. A file whose extension and content do not match (e.g. text file renamed to .pdf)."""
        fake_pdf = b"Plain text file that is masquerading as a PDF file."
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(fake_pdf), "fake.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertIn("<strong>Error:</strong>", html)
        self.assertIn("File content does not match the .pdf extension", html)

    def test_10_reupload_same_file_after_failed_attempt(self):
        """10. Re-uploading the same file after a failed attempt."""
        # 1. Attempt failed upload with bad file
        bad_resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(b"bad"), "test.xyz", "application/octet-stream")},
            content_type="multipart/form-data"
        )
        self.assertIn("<strong>Error:</strong>", bad_resp.data.decode("utf-8", errors="ignore"))
        self.assertIn("Unsupported file format", bad_resp.data.decode("utf-8", errors="ignore"))

        # 2. Upload valid PDF
        pdf_bytes = create_test_text_pdf()
        good_resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(pdf_bytes), "valid_document.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(good_resp.status_code, 200)
        html = good_resp.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html)
        self.assertIn("resultsSection", html)

    def test_11_repeated_valid_uploads_without_server_restart(self):
        """11. Repeated valid uploads without server restart (5 in a row)."""
        pdf_bytes = create_test_text_pdf()
        for i in range(5):
            resp = self.client.post(
                "/",
                data={"file": (io.BytesIO(pdf_bytes), f"consecutive_{i}.pdf", "application/pdf")},
                content_type="multipart/form-data"
            )
            self.assertEqual(resp.status_code, 200)
            html = resp.data.decode("utf-8", errors="ignore")
            self.assertNotIn("<strong>Error:</strong>", html)
            self.assertIn("resultsSection", html, f"Failed on consecutive upload #{i+1}")

    def test_12_oversized_upload(self):
        """12. An oversized upload (> 16MB)."""
        oversized_data = b"0" * (17 * 1024 * 1024)
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(oversized_data), "huge_file.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertIn(resp.status_code, [302, 413])

    def test_13_upload_failure_followed_by_another_successful_upload(self):
        """13. Upload failure followed by another successful upload."""
        # 1. Failure: empty file
        f_resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(b""), "blank.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertIn("<strong>Error:</strong>", f_resp.data.decode("utf-8", errors="ignore"))
        self.assertIn("empty (0 bytes)", f_resp.data.decode("utf-8", errors="ignore"))

        # 2. Success: valid JPG
        jpg_bytes = create_test_image("JPEG")
        s_resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(jpg_bytes), "subsequent_success.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(s_resp.status_code, 200)
        html = s_resp.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html)
        self.assertIn("resultsSection", html)

    def test_14_tamil_filename_handling(self):
        """14. Filename containing Tamil / Unicode characters."""
        pdf_bytes = create_test_text_pdf()
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(pdf_bytes), "பட்டா_ஆவணம்_2026.pdf", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html)
        self.assertIn("resultsSection", html)
        self.assertIn("பட்டா_ஆவணம்_2026.pdf", html)

    def test_15_pdf_with_octet_stream_mime(self):
        """15. Valid PDF sent with application/octet-stream MIME type."""
        pdf_bytes = create_test_text_pdf()
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(pdf_bytes), "binary_mime.pdf", "application/octet-stream")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html)
        self.assertIn("resultsSection", html)

    def test_16_filename_with_spaces_and_trailing_whitespace(self):
        """16. Filename with spaces and trailing whitespace."""
        pdf_bytes = create_test_text_pdf()
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(pdf_bytes), "  Government Patta Doc 2026 .pdf  ", "application/pdf")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="ignore")
        self.assertNotIn("<strong>Error:</strong>", html)
        self.assertIn("resultsSection", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
