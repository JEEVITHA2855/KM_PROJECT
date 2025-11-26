# KMRL Pure ML Alert Analysis System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)

**Single-file multilingual alert analysis system powered by pure ML models.**

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch transformers sentence-transformers scikit-learn numpy pandas

# Run the system
python kmrl_analyzer.py --text "Emergency brake failure" --json
```

## ✨ Features

- **🤖 Pure ML**: 100% transformer-based classification
- **🌍 Multilingual**: Works with 100+ languages
- **📦 Single File**: Everything in `kmrl_analyzer.py`
- **🎯 Multi-Label**: Severity + Alert Type + Department
- **🔍 Smart NER**: Extracts deadlines, amounts, regulations
- **⚡ Fast**: Real-time processing with confidence scores

## 📊 Example Output

```json
{
  "alert_id": "KMRL_ML_20251124_162028",
  "severity": "HIGH",
  "department": "SAFETY", 
  "alert_type": "SAFETY",
  "confidence": 82.5,
  "priority": "P1_CRITICAL",
  "search_tags": ["brake", "emergency", "safety"],
  ## Important Notes

  - The tool no longer reports internal processing timing fields (e.g., `processing_time_ms`).
  - The `--benchmark` option and timing output were removed to keep JSON output minimal and consistent for downstream systems.

  ## 📊 Example Minimal JSON Output

  ```json
  {
    "alert_id": "KMRL_ML_20251126_133048",
    "severity": "INFORMATIONAL",
    "department": "SAFETY",
    "alert_type": "SAFETY",
    "confidence": 55.7,
    "priority": "P4_LOW",
    "search_tags": ["tech_brake"],
    "immediate_action": false,
    "response_time": "24 hours",
    "timestamp": "2025-11-26 13:30:48",
    "model_used": "pure_ml",
    "ml_available": true,
    "multilingual": true
  }
  ```

  ## 💻 Usage

  ```bash
  # Interactive mode
  python kmrl_analyzer.py

  # Direct text analysis
  python kmrl_analyzer.py --text "Emergency brake failure detected" --json

  # Minimal output (compact fields)
  python kmrl_analyzer.py --text "Emergency brake failure" --minimal --json

  # File processing
  python kmrl_analyzer.py --file alerts.txt --json

  # Batch processing
  python kmrl_analyzer.py --batch
  ```

  ## ⚡ Fast vs Accurate Modes

  - By default the analyzer runs in **accurate** mode (larger multilingual models).
  - Use **fast** mode for lower memory and faster inference (smaller models + rule-based NER).

  Set via environment variable (PowerShell):
  ```powershell
  $env:KMRL_FAST_MODE = "true"
  python kmrl_analyzer.py --text "Example alert" --json
  ```

  Or use the CLI flag (temporary for that run):
  ```bash
  python kmrl_analyzer.py --fast --text "Example alert" --json
  ```