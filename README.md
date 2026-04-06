# KMRL Alert Analysis System

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Operational alert analysis tools for classifying incident text, routing alerts to the right department, and finding semantically similar cases. The repository contains two runnable analyzers:

- [analyzer.py](analyzer.py) for the scikit-learn + TF-IDF workflow
- [kmrl_analyzer.py](kmrl_analyzer.py) for the transformer-based workflow

## Overview

The project turns free-form operational alerts into structured, actionable output. It is designed for teams that need faster triage, more consistent routing, and lightweight semantic search without introducing a heavy web application stack.

## Problem Statement

Operational teams often receive short, inconsistent, or duplicated incident messages. Manual triage takes time, and keyword-only filtering misses related alerts that use different wording.

## Solution

The analyzers convert alert text into machine-readable signals, classify severity and department, extract keywords, and support similarity search. This produces a consistent output format that is easier to review, automate, and integrate into downstream workflows.

## Features

- Severity classification: CRITICAL, HIGH, MEDIUM, LOW
- Department routing: SAFETY, OPERATIONS, MAINTENANCE, FINANCE, HR, LEGAL, PROCUREMENT, ELECTRICAL
- Semantic search across previously seen alerts
- Keyword extraction for compact summaries
- Single-alert, file-based, batch, and interactive CLI modes
- JSON output for scripts and automation

## Tech Stack

- Frontend: None; command-line interface
- Runtime: Python 3.8+
- Core libraries: argparse, logging, json, pathlib
- ML / NLP: scikit-learn, TF-IDF, Multinomial Naive Bayes, transformers, sentence-transformers, PyTorch, NumPy, SciPy, tqdm
- Storage: In-memory during execution

## Architecture / Flow

1. A user submits alert text through the CLI, a file, or interactive mode.
2. The analyzer normalizes the text and converts it into numeric features or embeddings.
3. The model predicts severity and department, then computes confidence scores.
4. The system extracts keywords or similarity matches to add context.
5. The CLI prints a readable report or JSON payload for downstream use.

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

4. Run the default analyzer:

```bash
python analyzer.py --text "Emergency brake failure"
```

5. Optional: run the alternate transformer-based analyzer:

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

### Help

```bash
python analyzer.py --help
python kmrl_analyzer.py --help
```

## Output Example

### Alert Classification
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
