# 🇨🇭 Zurich Corporate Credit Risk Intelligence (MLOps Pipeline)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-green.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready machine learning pipeline for corporate credit risk assessment, built for the financial hub of Zurich, adhering to strict regulatory and risk management standards.**

---

## 🎯 Executive Summary & Business Problem

In the competitive financial landscape of Zurich, corporate banks face increasing pressure to balance aggressive lending growth with stringent regulatory compliance (**Basel III/IV guidelines** and **FINMA** frameworks). Evaluating Small and Medium-sized Enterprises (SMEs) for commercial loans is traditionally slow, manual, and prone to human bias or delayed risk detection.

### Core Business Questions Addressed:
1. **Predictive Indicators:** Which combination of financial health indicators (liquidity ratios, debt-to-equity, operational cash flows) acts as an early warning sign for corporate default within a 12-month horizon?
2. **Automation & Risk Mitigation:** How can we automate initial credit decisions while strictly constraining the **False Negative Rate below 5%** (preventing high-risk defaulting clients from being misclassified as creditworthy)?

---

## 🧰 Technology Stack

* **Language:** Python 3.10+
* **IDE:** Visual Studio Code (VS Code)
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Visualization:** Seaborn, Matplotlib
* **MLOps & Experiment Tracking:** MLflow
* **Version Control:** Git & GitHub

---

## 📁 Project Structure

The repository is modularly organized following enterprise software engineering best practices:

```text
zurich-credit-risk/
│
├── data/                      # Raw and processed corporate financial data
├── notebooks/                 # Exploratory Data Analysis (EDA) in VS Code
├── src/                       
│   ├── __init__.py
│   ├── preprocess.py          # Data cleaning and feature engineering
│   ├── train.py               # Model training and hyperparameter tuning
│   └── evaluate.py            # Performance evaluation and metrics logging
├── mlruns/                    # MLflow tracking artifacts and experiment logs
├── .gitignore                 # Excluded system, cache, and virtual environments
├── LICENSE                    # MIT Open Source License
├── requirements.txt           # Pinned project dependencies
└── README.md                  # Project documentation (this file)

---

## 📈 Results -  Business Conclusions
Three distinct machine learning models were trained and benchmarked under strict cross-validation protocols: Logistic Regression, Random Forest, and XGBoost.

Performance Leader: XGBoost outperformed baseline models, achieving an AUC-ROC of 0.89, successfully capturing non-linear distress patterns in corporate balance sheets.

Risk Control: The optimized classification threshold kept the False Negative Rate under 4.2%, safeguarding the bank’s capital reserves against unexpected defaults.

Strategic Value: Automating the preliminary scoring pipeline reduces corporate loan review turnaround times from several business days to under 5 minutes, drastically improving operational efficiency and client experience without elevating risk exposure.

---

🚀 Getting Started & Prerequisites
To run this project locally on your machine using VS Code, follow these step-by-step instructions:

Prerequisites
Ensure you have Python 3.10+ installed.

Ensure Git is installed on your system.

## 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

## Author
Robinson Alves Dos Santos
Data Scientist, Machine Learning Engineer, Physicist, PhD
