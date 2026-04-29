# KMRL Alert Analysis System

A machine learning–based system for analyzing operational alerts, classifying severity, routing incidents to the appropriate department, and identifying semantically similar cases. The system is designed for fast triage, consistent decision-making, and integration into operational workflows.

---

## Overview

The KMRL Alert Analysis System transforms unstructured alert text into structured, actionable outputs. It enables teams to automate alert classification, reduce manual triage effort, and improve response time by providing consistent severity predictions, department routing, and contextual insights.

---

## Problem Statement

Operational systems generate large volumes of alerts that are:

* Unstructured and inconsistent
* Difficult to categorize quickly
* Often duplicated or semantically similar

Manual triage leads to delays, inconsistent decisions, and missed correlations between related alerts.

---

## Solution

This system provides a lightweight analysis pipeline where:

* Alert text is processed and normalized
* Machine learning models classify severity and department
* Semantic similarity identifies related incidents
* Keywords are extracted for quick understanding
* Results are returned in structured JSON format for automation

---

## Key Features

* Severity classification (CRITICAL, HIGH, MEDIUM, LOW)
* Department routing (SAFETY, OPERATIONS, MAINTENANCE, etc.)
* Semantic similarity search across alerts
* Keyword extraction for summarization
* Multiple execution modes (CLI, batch, interactive)
* JSON output for integration with external systems

---

## Tech Stack

* Python 3.8+
* scikit-learn (TF-IDF, Naive Bayes)
* transformers, sentence-transformers
* PyTorch
* NumPy, SciPy
* argparse, logging, json

---

## System Architecture

```text id="9yx0k3"
Input (CLI / File / Interactive)
        ↓
Text Preprocessing
        ↓
Feature Extraction (TF-IDF / Embeddings)
        ↓
Classification (Severity + Department)
        ↓
Semantic Similarity Search
        ↓
Keyword Extraction
        ↓
Structured Output (JSON / CLI Report)
```

---

## Model Approaches

The system supports two analysis pipelines:

### 1. TF-IDF + Naive Bayes

* Lightweight and fast
* Suitable for real-time classification
* Lower computational cost

### 2. Transformer-Based Pipeline

* Uses sentence embeddings
* Better semantic understanding
* Higher accuracy for complex alerts

---

## Example Workflow

* User inputs alert text
* System preprocesses and extracts features
* Model predicts severity and department
* Similar alerts are retrieved
* Keywords are extracted
* Output is displayed or returned as JSON

---

## Example Output

```json id="z6n1qp"
{
  "alert_id": "KMRL_20251202_153842",
  "text": "Emergency brake failure",
  "severity": "CRITICAL",
  "department": "SAFETY",
  "priority": "P1_CRITICAL",
  "response_time": "5 minutes",
  "keywords": ["brake", "failure", "emergency"],
  "immediate_action": true
}
```

---

## Setup Instructions

### Prerequisites

* Python 3.8+
* pip

### Installation

```bash id="q9k4fm"
git clone https://github.com/JEEVITHA2855/KMRL-Alert-Analysis-System.git
cd KM_PROJECT
python -m pip install -r requirements.txt
```

---

## Usage

### Analyze a Single Alert

```bash id="zps2lu"
python analyzer.py --text "Emergency brake failure"
```

### JSON Output

```bash id="y4mjv3"
python analyzer.py --text "Emergency brake failure" --json
```

### Semantic Search

```bash id="a6lp7r"
python analyzer.py --search "fire danger"
```

### Interactive Mode

```bash id="2o0n1w"
python analyzer.py -i
```

### Batch Processing

```bash id="7b7k0n"
python analyzer.py --batch --file alerts.txt
```

---

## Engineering Highlights

* Designed dual ML pipelines for performance vs accuracy trade-offs
* Implemented text classification using TF-IDF and Naive Bayes
* Integrated transformer-based embeddings for semantic similarity
* Built modular CLI system supporting multiple execution modes
* Structured outputs for easy integration with external systems

---

## Challenges

* Balancing lightweight models with transformer-based accuracy
* Managing model loading and performance constraints
* Ensuring consistent outputs across different execution modes
* Handling variability in unstructured alert text

---

## Future Improvements

* Add persistent storage for alert history
* Build a web-based dashboard for visualization
* Improve training data and evaluation metrics
* Add role-based workflows for incident management
* Introduce real-time alert streaming

---

## Resume Impact

Developed a machine learning–based alert classification system using TF-IDF, Naive Bayes, and transformer models, enabling automated incident triage, semantic similarity search, and structured output for operational workflows.

---

## Notes

* Designed for local execution and integration into larger systems
* Supports both lightweight and transformer-based analysis pipelines
* Focuses on practical applicability in operational environments

---
