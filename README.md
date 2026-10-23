# GovEase AI — Bilingual Government Document Understanding and Scheme Recommendation Assistant

An advanced, bilingual (English & தமிழ்) AI-powered assistant designed to help citizens comprehend complex government documents (certificates, court orders, land records, circulars) and discover relevant public welfare schemes.

---

## 🏛️ Key Features

1. **Universal Document Understanding**:
   - Handles images (`.png`, `.jpg`, `.jpeg`) and multi-page PDFs (both text-based and scanned).
   - Classifies 15+ government document types: First Graduate Certificates, Income & Community Certificates, Court Orders & Circulars, Patta / Chitta & Land Records, Legal Heir Certificates, Birth & Death Certificates, Ration Cards, Aadhaar (with 8-digit privacy masking), and General Government Documents.
   - **General Government Document Fallback**: Never rejects a readable government document simply because its specific certificate category is unknown.

2. **Advanced OpenCV + Tesseract OCR Pipeline**:
   - Supports English (`eng`) and Tamil (`tam`).
   - Automatic perspective correction & quad contour document detection.
   - Intelligent deskewing & text angle alignment.
   - Multiple image enhancement variants (CLAHE contrast equalization, Gaussian adaptive thresholding, bilateral noise filtering, sharpening).
   - Multi-PSM trial strategy (PSM 3, 4, 6, 11) to pick the highest confidence extraction.
   - Quality confidence scoring (`Good`, `Moderate`, `Low`) with transparent user warnings.

3. **Universal AI-Powered Summarizer**:
   - **3 Distinct Modes**:
     - ⚡ **Quick Summary**: 3–5 easy-to-understand bullet points.
     - 📖 **Detailed Summary**: Structured explanation of purpose, terms, and context.
     - 🎯 **Next Action Points**: Concrete next steps required from the citizen (e.g. college admissions counseling, court appearance, document preservation).
   - Generates output in both **English and Tamil (தமிழ்)**.
   - Identifies critical entities: Issuing authorities, reference numbers, monetary amounts, and key dates.
   - Exports summaries to professional bilingual **PDF Reports** and plain text files.

4. **Government Scheme Recommendation Engine**:
   - Accessible independently via `/schemes/`.
   - Categories: Agriculture, Education, Women & Child, Skills & Employment, Business & Entrepreneurship, Financial Assistance, Housing, Health & Wellness, Senior Citizens, Differently Abled, and General Public Welfare.
   - Calibrated percentage match scores (`0–100%`) with transparent **"Why it matched"** reasons.
   - Scheme comparison drawer (compare up to 3 schemes side-by-side) and local saving.
   - Official source links and prominent *"Possible Match — Verify Official Eligibility"* disclaimers.

5. **Secure Authentication & Document History**:
   - Secure account registration & login with password hashing.
   - Private document history: logged-in users have their uploaded documents and summaries saved automatically.
   - User isolation: users can only view and manage their own history.
   - Document review: revisit any past document analysis and re-download its PDF report.

---

## 💻 System Requirements & Prerequisites (Windows)

- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Python**: Python 3.10 to 3.14
- **Tesseract OCR**: 5.x installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - Language Data: `eng.traineddata` (English) and `tam.traineddata` (Tamil)

---

## 📦 Tesseract OCR & Tamil Language Data Setup

1. **Download and Install Tesseract OCR for Windows**:
   - Download the Windows 64-bit installer from UB-Mannheim:
     [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - Run the installer and install to the default location:
     `C:\Program Files\Tesseract-OCR`
   - During installation, check the box under **Additional language data** for **Tamil**.

2. **If Tamil data is missing (`tam.traineddata`)**:
   - Download `tam.traineddata` from the official Tesseract tessdata repository:
     [https://github.com/tesseract-ocr/tessdata/raw/main/tam.traineddata](https://github.com/tesseract-ocr/tessdata/raw/main/tam.traineddata)
   - Copy `tam.traineddata` into:
     `C:\Program Files\Tesseract-OCR\tessdata\`

3. **Verify Installation**:
   Open PowerShell and run:
   ```powershell
   & "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
   ```
   Ensure both `eng` and `tam` appear in the output list.

---

## 🚀 Installation & Running the Project

1. **Open PowerShell in the Project Directory**:
   ```powershell
   cd "c:\Users\tharu\DOCUMENT ASSISTENCE"
   ```

2. **Activate the Virtual Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies** (if needed):
   ```powershell
   pip install -r requirements.txt
   ```

4. **Start the Flask Application**:
   ```powershell
   python app.py
   ```

5. **Open in Web Browser**:
   Navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## 🧪 Testing the Application

Run the automated verification suite to test all components:

```powershell
python -m py_compile app.py auth_system.py scheme_recommendation.py summarizer.py ocr_pipeline.py document_classifier.py court_processor.py pdf_exporter.py readability_analyzer.py
```

### Manual Test Scenarios

| Test Case | Steps | Expected Result |
| :--- | :--- | :--- |
| **1. User Registration** | Go to `/register`, enter Name, Email, Password. | Account created and redirected to `/`. |
| **2. User Login** | Go to `/login`, enter registered credentials. | Logged in successfully; header displays user name. |
| **3. Document Upload (Image)** | Upload a certificate image (e.g. First Graduate). | Document classified, bilingual summaries (Quick, Detailed, Action Points), fields extracted, image preview shown. |
| **4. PDF Support** | Upload a multi-page or scanned PDF document. | PDF parsed, text extracted across pages, first-page preview generated. |
| **5. PDF Summary Export** | Click **"📄 Download PDF"** on the analysis page. | Downloads bilingual PDF report with English & Tamil text. |
| **6. Document History** | Navigate to `/history`. | Lists uploaded document with OCR confidence. Click **View Summary** to re-open. |
| **7. Scheme Recommendation** | Navigate to `/schemes/`, pick a category, and click **Find Matching Schemes**. | Displays matching schemes with match %, "Why it matched" reasons, and official links. |
| **8. Language Toggle** | Click the **"தமிழ்"** button in the header. | Interface labels, summaries, action points, and fields switch to Tamil instantly. |

---

## 🗂️ Project Directory Structure

```
DOCUMENT ASSISTENCE/
├── app.py                      # Core Flask application and routing
├── auth_system.py              # User authentication & document history management
├── court_processor.py          # Legal entity extraction & court order parser
├── document_classifier.py      # Universal document classifier & field extractor
├── ocr_pipeline.py             # Preprocessing, deskewing, OpenCV & Tesseract OCR
├── pdf_exporter.py             # PyMuPDF-based bilingual PDF summary report generator
├── readability_analyzer.py     # Flesch Reading Ease & text complexity metrics
├── scheme_recommendation.py    # Public scheme matching engine with in-memory caching
├── schemes.csv.xls             # Public welfare scheme dataset (~19.5 MB)
├── users.db                    # SQLite database (users and document_history tables)
├── requirements.txt            # Python package dependencies
├── .env.example                # Example environment configuration
├── templates/
│   ├── index.html              # Main dashboard, upload & document analysis page
│   ├── history.html            # User document history page
│   ├── account.html            # User account profile & stats
│   ├── login.html              # User login portal
│   ├── register.html           # User registration portal
│   └── schemes.html            # Government scheme recommendation portal
├── uploads/                    # Secure folder for uploaded documents
└── processed/                  # Folder for processed preview images
```

---

## 🔒 Security & Privacy Features

- **Safe Filenames**: Uses `secure_filename()` with unique timestamps to prevent directory traversal and name collisions.
- **Aadhaar Privacy**: Automatically masks the first 8 digits of Aadhaar numbers (`XXXX XXXX 1234`).
- **User Isolation**: SQLite queries enforce strict `user_id` boundaries so users cannot access each other's documents.
- **Password Security**: Strong hashing with `werkzeug.security` (scrypt / pbkdf2).
- **Safe Output Rendering**: All OCR and user-supplied strings are escaped in Jinja2 templates.
