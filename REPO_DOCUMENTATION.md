# KMRL Alert Detection — Repository Documentation

Last updated: 2025-11-19

This document is the canonical repository documentation for the KMRL Alert Detection demo system. It summarizes what was built, the exact tech stack and models used, how we trained and evaluated models, how to run the demo and web UI locally, and recommended next steps for production.

---

## Project overview (short)
A complete demo-ready system that automatically detects "alerts" in KMRL documents using ML instead of hardcoded stopwords. It classifies document severity (Critical/High/Medium/Low) and assigns a department (Safety, Operations, Finance, HR). Includes:
- Data: sample labeled dataset
- Preprocessing: text cleaning and TF-IDF feature extraction
- Models: Logistic Regression (severity) and Random Forest (department)
- Interfaces: CLI demo, Jupyter notebook, and a Flask web UI with analytics

This repository is intended as an MVP/demo for stakeholders and as a starting point for productionization.

---

## Repo layout (files & purpose)
```
KM_PROJECT/
├── data/
│   ├── sample_kmrl_documents.csv          # 40 labeled sample documents used for training/demo
│   └── labeling_guidelines.md             # Labeling instructions and template
├── scripts/
│   ├── preprocessing.py                   # Text cleaning + preprocessing utilities
│   ├── train_model.py                     # Training pipeline (TF-IDF + models) and save/load methods
│   └── demo.py                            # CLI interactive demo used for quick testing
├── models/                                # Trained model artifacts (auto-created by training)
│   ├── severity_model.pkl
│   ├── department_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── preprocessor.pkl
├── notebooks/
│   └── KMRL_Alert_Detection_Demo.ipynb    # Presentation notebook (visualizations & training code)
├── web_app/
│   ├── app.py                             # Flask backend and REST endpoints
│   └── templates/                         # HTML templates: index.html, analytics.html, comparison.html
├── README.md                              # Quick start / summary (smaller)
├── DEMO_SUMMARY.md                        # Demo summary used for stakeholders
├── WEB_UI_GUIDE.md                        # Web UI guide and steps
├── REPO_DOCUMENTATION.md                  # <-- This file (comprehensive documentation)
└── requirements.txt                       # Python dependencies
```

---

## Tech stack (exact)
- Language: Python 3.12
- ML & Data: scikit-learn (LogisticRegression, RandomForestClassifier, TfidfVectorizer), pandas, numpy
- Persistence: joblib for saving models
- Web: Flask (templates with Bootstrap 5, JS + Chart.js for charts)
- Notebooks: Jupyter
- Visualization: matplotlib, seaborn (used in notebook)

Dependencies are listed in `requirements.txt` (install with pip).

---

## Models used (exact)
1. TF-IDF feature extractor
   - `sklearn.feature_extraction.text.TfidfVectorizer`
   - Parameters used: max_features=5000, stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.8

2. Severity classifier
   - `sklearn.linear_model.LogisticRegression`
   - Purpose: multi-class classification for [Critical, High, Medium, Low]
   - Params: random_state=42, max_iter=1000

3. Department classifier
   - `sklearn.ensemble.RandomForestClassifier`
   - Purpose: classify department (Safety, Operations, Finance, HR)
   - Params: n_estimators=100, random_state=42

4. Label encoders
   - `sklearn.preprocessing.LabelEncoder` used to map string labels ↔ integers

Model artifacts are saved with joblib in `models/`.

---

## Data (summary)
- File: `data/sample_kmrl_documents.csv`
- 40 sample documents across 4 severity labels and 4 departments. Realistic, domain-specific examples were created to demonstrate behavior.
- Use the labeling guidelines in `data/labeling_guidelines.md` for adding more labeled documents.

Notes: This dataset is intentionally small for demo. For production-level accuracy, collect hundreds to thousands of labeled examples per class.

---

## Preprocessing summary
- Text cleaning performed in `scripts/preprocessing.py` (TextPreprocessor):
  - convert to lowercase
  - remove special characters except whitespace
  - collapse multiple spaces
  - remove tokens shorter than 2 characters
- Labels encoded using LabelEncoder for both severity and department
- TF-IDF used to convert cleaned_text → numerical features

---

## Training (exact steps we used)
The training pipeline is in `scripts/train_model.py`. Summary of the process:
1. Load `data/sample_kmrl_documents.csv` and preprocess (clean text + encode labels)
2. Prepare features `X = cleaned_text`, targets `y_severity` and `y_department`
3. Split: `train_test_split(..., test_size=0.2, stratify=y_severity, random_state=42)`
4. Fit TF-IDF on training data and transform both train/test
5. Train severity model: LogisticRegression
6. Train department model: RandomForestClassifier
7. Evaluate using accuracy and classification_report on the test split
8. Save models: `joblib.dump(...)` into `models/`

To run training locally (PowerShell):
```powershell
cd e:\KM_PROJECT\scripts
C:/Python312/python.exe train_model.py
```
Outputs: printed evaluation metrics and saved model files in `models/`.

---

## Demo & UI (how to run)
### 1) Install dependencies
```powershell
cd e:\KM_PROJECT
C:/Python312/python.exe -m pip install -r requirements.txt
```
(You may already have some packages installed. If not, use the above.)

### 2) Start web UI (Flask)
```powershell
cd e:\KM_PROJECT\web_app
C:/Python312/python.exe app.py
```
Open: http://localhost:5000

### 3) CLI demo
```powershell
cd e:\KM_PROJECT\scripts
C:/Python312/python.exe demo.py
```
This runs an interactive CLI demo that processes a few sample documents and compares results to a stopword approach.

### 4) Jupyter Notebook demo
```powershell
cd e:\KM_PROJECT\notebooks
jupyter notebook KMRL_Alert_Detection_Demo.ipynb
```

---

## API endpoints (used by web UI)
- `GET /` — dashboard page (index)
- `POST /predict` — accepts JSON {"text": "..."} and returns prediction JSON with fields: severity, department, severity_confidence, department_confidence, alert_required, timestamp
- `GET /demo_documents` — returns sample demo documents (used to populate quick test buttons)
- `GET /analytics` — analytics page showing aggregated alert history
- `GET /comparison` — page that explains difference vs stopwords

---

## What we tested/validated
- Training pipeline runs end-to-end on the sample dataset and saves models
- Demo scripts (CLI and notebook) run and display predictions and evaluation metrics
- Flask web UI runs locally, loads model artifacts, performs real-time predictions, and displays analytics and charts

---

## Limitations & recommendations for production
- Dataset size: sample set is small (40 docs). Collect and label more KMRL documents (500–10,000+) to improve performance.
- Model choice: TF-IDF + LogisticRegression + RandomForest is good for MVP. For better context and multilingual (Malayalam) support, fine-tune a multilingual transformer (XLM-RoBERTa / IndicBERT).
- Scalability: Move from Flask dev server to production stack (Gunicorn + Nginx) or FastAPI + Uvicorn, Docker, and Kubernetes for scaling.
- Storage: Move from CSV + joblib -> PostgreSQL / object storage and model registry (MLflow).
- Logging & Monitoring: Add structured logging, Prometheus/Grafana, and health checks.
- OCR: For scanned images/PDFs, integrate an OCR step (Tesseract or cloud OCR) before preprocessing.

---

## Next steps (recommended roadmap)
1. Collect labeled production documents, run active learning, expand dataset.
2. Train a transformer (XLM-R / IndicBERT) with domain-specific fine-tuning for better accuracy and Malayalam support.
3. Add authentication & RBAC to the UI and REST API.
4. Persist alert history into a database for long-term analytics.
5. Deploy with Docker + Kubernetes and add load testing.
6. Implement CI/CD (tests, linting, training pipeline automation) and model versioning (MLflow).

---

## Exact commands used in demo environment (PowerShell)
```powershell
# Install core deps
cd e:\KM_PROJECT
C:/Python312/python.exe -m pip install -r requirements.txt

# Train models (if you want to re-train)
cd e:\KM_PROJECT\scripts
C:/Python312/python.exe train_model.py

# Run CLI demo
C:/Python312/python.exe demo.py

# Start web UI
cd e:\KM_PROJECT\web_app
C:/Python312/python.exe app.py
```

---

## Where to find the important code
- `scripts/preprocessing.py` — text cleaning and label encoding
- `scripts/train_model.py` — training, evaluation, save/load of models
- `scripts/demo.py` — CLI demo and comparison to stopwords
- `web_app/app.py` — Flask web server and endpoints
- `web_app/templates/` — UI pages (index, analytics, comparison)
- `notebooks/KMRL_Alert_Detection_Demo.ipynb` — notebook used for interactive demo and plots

---

## Contact & authorship
- Author: code and demo created in this workspace for KMRL demo
- For integration help, productionization, or adding transformer-based models, create an issue in the repository and assign a priority.

---

## License
Add your preferred license file (e.g., `LICENSE`) at repo root. This repo does not include a license by default — add one before sharing publicly if required.

---

## Changelog (quick)
- 2025-09-15 — Project scaffold, sample data, preprocessing and baseline models
- 2025-11-19 — Demo web UI, analytics pages, README and full repo documentation

---

If you want, I can:
- Commit this file to the repo (`e:/KM_PROJECT/REPO_DOCUMENTATION.md`) and open a short PR-style commit message.
- Add a small `CONTRIBUTING.md` and `LICENSE` file.
- Add a `docs/` folder and split long docs into separate pages.

Tell me which of those you'd like next and I'll do it for you.