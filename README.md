# ACME Customer Happiness Predictor (Apziva Project)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![Framework](https://img.shields.io/badge/Frontend-Streamlit-red)
![Framework](https://img.shields.io/badge/Backend-Flask-green)
![Deployment](https://img.shields.io/badge/Deploy-Docker%20%7C%20Hugging%20Face%20Spaces-black)

Predict whether a customer is **Happy (1)** or **Unhappy (0)** using survey responses from a logistics and delivery startup (ACME).  
This project covers the full ML lifecycle: **EDA → modeling → feature selection → evaluation → deployment as a web app**.

---

## Table of Contents
- [Project Summary](#project-summary)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Solution Overview](#solution-overview)
- [Step-by-Step Project Flow](#step-by-step-project-flow)
- [Results](#results)
- [Application Architecture](#application-architecture)
- [Repository Structure](#repository-structure)
- [How to Run Locally](#how-to-run-locally)
- [Run with Docker](#run-with-docker)
- [Deployment Notes (Hugging Face Spaces)](#deployment-notes-hugging-face-spaces)
- [Key Takeaways](#key-takeaways)
- [Future Improvements](#future-improvements)
- [Concluding Note for Hiring Managers](#concluding-note-for-hiring-managers)

---

## Project Summary
ACME collects customer survey feedback, but needs a scalable way to **predict satisfaction** and understand what factors most influence happiness.  
I built and evaluated multiple classification models and shipped the best-performing approach as a simple **Streamlit + Flask** application that can be deployed via **Docker**.

---

## Business Problem
ACME is expanding rapidly and needs a reliable method to:
1. Predict customer happiness from survey answers.
2. Identify the most important survey questions to reduce survey length and improve signal quality.

**Success metric:** Achieve **≥ 73% accuracy**, or provide a strong solution that delivers actionable insights.

---

## Dataset
The dataset contains **126 survey responses** with the following columns:

- **Target**
  - `Y`: customer happiness (`0 = unhappy`, `1 = happy`)

- **Features (1–5 Likert scale)**
  - `X1`: My order was delivered on time  
  - `X2`: Contents of my order was as I expected  
  - `X3`: I ordered everything I wanted to order  
  - `X4`: I paid a good price for my order  
  - `X5`: I am satisfied with my courier  
  - `X6`: The app makes ordering easy for me  

---

## Solution Overview
### Models explored
- Logistic Regression (statsmodels + interpretability + feature significance)
- Decision Tree (baseline + tuned)
- Random Forest (baseline + tuned)
- SVM
- Naive Bayes

### Key decision
Early analysis showed some features contributed little to predictive power.  
I tested a **reduced feature set** and re-trained models to improve generalization.

✅ Final model: **Random Forest using reduced features:** `X1, X2, X3, X5`

---

## Step-by-Step Project Flow

### 1) Data Loading & Quality Checks
- Loaded survey data
- Checked shape, data types, duplicates, and missing values
- Confirmed no null values and handled duplicates

### 2) Exploratory Data Analysis (EDA)
- Univariate distributions (histograms + boxplots)
- Correlation analysis (heatmap)
- Feature vs target analysis using stacked and distribution plots

**Insight:** `X1`, `X5`, `X3` (and sometimes `X6`) appeared more correlated with happiness than `X2` and `X4`.

### 3) Outlier Handling
- Used boxplots for detection
- Applied flooring/capping (`np.clip`) to reduce outlier distortion

### 4) Modeling & Evaluation
- Train/test split (80/20)
- Baseline modeling across multiple algorithms
- Identified overfitting patterns (especially tree-based models on small data)

### 5) Tuning & Feature Selection
- Logistic regression feature elimination using p-values (interpretability step)
- GridSearchCV for tree-based models to reduce overfitting

### 6) Feature Reduction & Re-Training
- Dropped weaker features (`X4`, `X6`)
- Re-ran training + evaluation across models
- Selected best-performing model based on accuracy and balanced precision/recall

### 7) Packaging & Deployment
- Saved trained model (`joblib`)
- Built:
  - **Flask API** for predictions
  - **Streamlit UI** for user interaction
- Dockerized the full solution with an entrypoint that runs both services

---

## Results
### Best Model (Random Forest — Reduced Features)
| Metric | Score |
|-------|------:|
| Accuracy | **0.730769** |
| Recall | 0.846154 |
| Precision | 0.687500 |
| F1-score | 0.758621 |

This model meets the project target (**≥ 73% accuracy**) while maintaining strong recall and a solid F1-score.

---

## Application Architecture

**User Flow**
1. User opens Streamlit UI
2. Inputs survey scores for `X1, X2, X3, X5`
3. Streamlit sends a POST request to Flask `/predict`
4. Flask loads the trained model and returns:
   - predicted class (happy/unhappy)
   - confidence score (probability)

**Why this architecture?**
- Separates concerns (UI vs model serving)
- Easy to scale backend independently
- Simple to deploy with Docker

---

## Repository Structure
├── README.md
├── data/
│ └── ACME-HappinessSurvey2020.csv
├── models/
│ └── happiness_prediction_model.joblib
├── notebooks/
│ ├── Happiness_Prediction.ipynb
│ └── Happiness_Prediction.html
└── src/
├── app.py # Streamlit frontend
├── backend.py # Flask prediction API
├── config.py # paths + selected features
├── predict.py # model loading + inference
├── Dockerfile
├── entrypoint.sh # runs backend + frontend
└── requirements.txt

## How to Run Locally

### Prerequisites
- Python 3.9+
- pip
- git

### Setup
```bash
git clone https://github.com/samuelmugisha/bSXsGeDhGRcNAdms.git
cd bSXsGeDhGRcNAdms/src
pip install -r requirements.txt
Run the backend (Flask)
python backend.py

Backend will run on: http://127.0.0.1:5000

Run the frontend (Streamlit)

Open a new terminal in the same folder and run:

streamlit run app.py

Frontend will open on: http://localhost:8501

Run with Docker

From the src/ folder:

docker build -t acme-happiness .
docker run -p 7860:7860 -p 5000:5000 acme-happiness

Streamlit UI: http://localhost:7860

Flask API: http://localhost:5000/predict

Deployment Notes (Hugging Face Spaces)

This repo is configured for Docker-based deployment on Hugging Face Spaces.

Key files:

Dockerfile

entrypoint.sh

requirements.txt

app.py, backend.py, config.py, predict.py

## Key Takeaways
- Feature reduction improved performance and reduced noise
- Small datasets amplify overfitting risk, making tuning essential
- Random Forest offered the best balance of metrics for this problem
- Deployment added real-world value beyond a notebook-only solution

## Future Improvements

- Collect more data to improve stability and reduce variance
- Add model monitoring (drift checks, logging, prediction analytics)
- Add explainability (SHAP) for decision transparency
- Improve calibration of confidence scores
- Expand UI with batch prediction and downloadable results

## Concluding Note for Hiring Managers

- This project demonstrates my ability to deliver an end-to-end machine learning solution:
translating a business objective into a measurable ML task, conducting EDA and model experimentation, applying feature selection and tuning to improve generalization,
and deploying the final model as a usable application (API + UI + Docker).

If you're hiring for roles involving applied machine learning, data science, or ML product delivery, I’d be excited to discuss the decisions, trade-offs, and results in more detail.
