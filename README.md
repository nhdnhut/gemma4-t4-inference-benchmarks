# Gemma-4 Latency Predictor and Inference Benchmark Suite

> An empirical benchmarking laboratory and Random Forest latency prediction harness for quantized Gemma models on NVIDIA T4 GPU hardware.
> Ollama Integration | Latency Predictor Training (Scikit-Learn) | Google Colab T4 Notebooks | Automated PDF Documentation Generator

[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-15%20Passed-brightgreen?style=flat-square)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

---

## 1. Overview

This laboratory suite investigates inference behavior, memory footprint, and token generation latency for quantized Gemma architectures on resource-constrained NVIDIA T4 GPU environments.

Key Components:
1. Three Modular Colab Notebooks:
   - `01_Baseline_Gemma4_T4.ipynb`: Environment initialization, Ollama service orchestration, digest locking, and baseline latency telemetry.
   - `02_Train_Optimize_Gemma4_T4.ipynb`: Random Forest inference latency predictor training based on empirical execution profiles across generation parameters.
   - `03_Evaluate_Demo_Gemma4_T4.ipynb`: Benchmark comparison against baseline controls, evaluation tables, and interactive QA.
2. Latency Predictor Architecture: Machine learning regression model (Random Forest Regressor) predicting end-to-end token latency based on context length, temperature, top-k/p, and hardware telemetry.
3. Automated Report Generation: Bundled Python script (`build_guide_pdf.py`) compiling technical documentation, Mermaid diagrams, and benchmark charts into printable PDF manuals.

---

## 2. Directory Structure

```
gemma4-t4-inference-benchmarks/
├── data/                       # Manual review templates and schema examples
├── docs/                       # Technical guide, Mermaid flowcharts, and audit logs
├── notebooks/                  # Reproducible Google Colab notebooks
│   ├── 01_Baseline_Gemma4_T4.ipynb
│   ├── 02_Train_Optimize_Gemma4_T4.ipynb
│   └── 03_Evaluate_Demo_Gemma4_T4.ipynb
├── original/                   # Reference notebooks for provenance tracking
├── results/                    # Documentation on expected output structures
├── src/                        # Core Python package
│   ├── controller.py           # Process orchestration and telemetry capture
│   ├── dataset.py              # Benchmark prompt generation and splitting
│   └── gemma_lab.py            # Latency predictor training and evaluation
├── tests/                      # Automated unit test suite (15 passed)
├── build_guide_pdf.py          # Automated PDF documentation generator
├── build_notebooks.py          # Self-contained notebook builder
├── validate_package.py         # Integrity and environment validator
├── requirements-colab.txt      # Dependency specification
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## 3. Quick Start Guide

### Prerequisites
- Python 3.10 or higher
- Google Colab account with access to T4 GPU runtimes

### Running Tests Locally
```bash
# Run unit tests on CPU
pytest tests/ -v
```

### Validating Package Integrity
```bash
python validate_package.py
```

### Running on Google Colab
1. Upload `notebooks/01_Baseline_Gemma4_T4.ipynb` to Google Colab.
2. Select Runtime -> Change runtime type -> T4 GPU.
3. Execute cells sequentially. Each notebook manages its own runtime state.

---

## 4. Author and License

- Author: Nhut Nguyen (@nhdnhut)
- License: MIT License (see [LICENSE](LICENSE))

