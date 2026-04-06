# KMRL Alert Analysis System

Lightweight Python tooling for classifying operational alerts, assigning departments, and performing semantic search for KMRL-style incident text. The repository contains two runnable analyzers:

- [analyzer.py](analyzer.py) for the scikit-learn + TF-IDF workflow
- [kmrl_analyzer.py](kmrl_analyzer.py) for the transformer-based workflow

## Overview

This project exists to help operations teams quickly interpret free-form alert text and turn it into a structured result. It predicts severity, routes alerts to a department, extracts useful keywords, and supports semantic search across previously seen alerts.

## Problem Statement

Operational teams receive short, inconsistent alert messages that are expensive to triage manually. The same incident can be described in many different ways, which makes keyword-only filtering unreliable and slows down response time.

## Solution

The project converts alert text into machine-readable signals and applies ML-based classification to produce a consistent result. It also supports semantic search so similar incident descriptions can be found even when they do not share the same exact words.

## Features

- Severity classification: CRITICAL, HIGH, MEDIUM, LOW
- Department routing: SAFETY, OPERATIONS, MAINTENANCE, FINANCE, HR, LEGAL, PROCUREMENT, ELECTRICAL
- Semantic search for similar alerts
- Keyword extraction for quick summaries
- CLI modes for single alert, file input, batch processing, and interactive use
- JSON output for automation and scripting

## Tech Stack

Frontend: None; this is a command-line Python project
Backend: Python 3.8+, argparse, logging
ML / NLP: scikit-learn, TF-IDF, Multinomial Naive Bayes, transformers, sentence-transformers, PyTorch, NumPy, SciPy, tqdm
Storage: In-memory during execution

## Architecture / Flow

1. User enters an alert text through the CLI, a file, or interactive mode.
2. The analyzer normalizes the text and converts it into numerical features.
3. The model predicts severity and department, then calculates confidence scores.
4. Keyword extraction or embeddings are used to summarize and compare alerts.
5. The CLI prints a readable report or JSON output for downstream use.

## Setup Instructions

### Prerequisites

- Python 3.8 or later
- `pip`
- Internet access on first run if model weights need to be downloaded

### Installation

1. Clone the repository.
2. Open a terminal in the `KM_PROJECT` folder.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Run the analyzer:

```bash
python analyzer.py --text "Emergency brake failure"
```

5. Optional: use the alternate transformer-based analyzer:

```bash
python kmrl_analyzer.py --text "Emergency brake failure"
```

## Environment Variables

Important optional variables used by the project:

```bash
KMRL_FAST_MODE=true
CUDA_VISIBLE_DEVICES=
```

- `KMRL_FAST_MODE=true` enables faster, lighter transformer settings in [kmrl_analyzer.py](kmrl_analyzer.py)
- `CUDA_VISIBLE_DEVICES=` forces CPU-only execution when needed

## Commands

### Analyze a Single Alert

```bash
python analyzer.py --text "Emergency brake failure"
```

### JSON Output

```bash
python analyzer.py --text "Emergency brake failure" --json
```

### Semantic Search

```bash
python analyzer.py --search "fire danger"
```

### Interactive Mode

```bash
python analyzer.py -i
```

### Batch Processing

```bash
python analyzer.py --batch --file alerts.txt
```

### Alternate Analyzer

```bash
python kmrl_analyzer.py --fast --text "Routine maintenance issue"
```

## Output Example

```json
{
    "alert_id": "KMRL_20251202_153842",
    "text": "Emergency brake failure",
    "severity": "CRITICAL",
    "department": "SAFETY",
    "priority": "P1_CRITICAL",
    "response_time": "5 minutes",
    "overall_confidence": 25.61,
    "keywords": ["brake", "failure", "emergency"],
    "immediate_action": true
}
```

## Deployment

This repository is designed for local execution and scripting rather than a hosted web deployment. Typical usage is:

- Local terminal execution for operators and analysts
- Scheduled or scripted batch runs for alert review
- Integration into a larger system by importing the analyzer classes directly

## Challenges Faced

- Balancing lightweight TF-IDF classification with a heavier transformer-based implementation
- Handling model downloads and environment setup on first run
- Keeping CLI output readable while still supporting JSON automation

## Future Improvements

- Persist alert history to a database or file store
- Add a web dashboard for operational teams
- Introduce role-based workflows for review and escalation
- Improve model training data and evaluation coverage
- Add automated tests for CLI modes and output formats

## Project Structure

```text
KM_PROJECT/
├── analyzer.py
├── kmrl_analyzer.py
├── requirements.txt
└── README.md
```

## Support

If dependencies fail to install or models fail to load, verify your Python version and rerun:

```bash
python --version
python -m pip install -r requirements.txt
```

The main entry point recommended for quick use is [analyzer.py](analyzer.py).
