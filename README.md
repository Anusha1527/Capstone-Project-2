# ❤️ Heart Disease Detection using Machine Learning  
### Capstone Project – Data Science Classification Challenge  

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

Heart disease is one of the leading causes of death globally. Early detection can significantly improve survival rates and reduce healthcare costs.

This project builds and optimizes multiple **Machine Learning classification models** to predict whether a patient has heart disease based on diagnostic test results and clinical measurements.

> 🎯 **Goal:** Accurately predict heart disease presence using clinical data and optimize model performance for real-world medical decision support.

---

## 🧠 Problem Statement

Can we accurately predict the presence of heart disease in patients using:

- Blood pressure  
- Cholesterol levels  
- ECG results  
- Exercise capacity  
- Clinical test indicators  

This project simulates a real-world healthcare analytics scenario where data scientists collaborate with medical professionals.

---

## 📊 Dataset Description

### 🎯 Target Variable
- `heart_disease`  
  - `0` → No Heart Disease  
  - `1` → Heart Disease Present  

### 📂 Feature Categories

#### 👤 Demographic
- `age`
- `sex`

#### 🏥 Clinical Measurements
- `chest_pain_type`
- `resting_blood_pressure`
- `cholesterol`
- `fasting_blood_sugar`

#### 🧪 Diagnostic Results
- `resting_ecg`
- `max_heart_rate`
- `exercise_induced_angina`
- `st_depression`
- `st_slope`
- `num_major_vessels`
- `thalassemia`

---

## 🔍 Project Workflow

### 📌 Phase 1: Exploratory Data Analysis (EDA)
- Data cleaning & preprocessing
- Missing value handling
- Correlation heatmaps
- Feature distribution visualization
- Train-Test split (80/20)

### 📌 Phase 2: Baseline Model Development
Implemented 4 classification algorithms:

- 🌳 Decision Tree
- 🌲 Random Forest
- 📈 Logistic Regression
- ⚡ Support Vector Machine (SVM)

Metrics Evaluated:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- ROC-AUC
- Confusion Matrix

### 📌 Phase 3: Hyperparameter Optimization
- GridSearchCV
- Stratified K-Fold Cross Validation
- Model performance comparison
- Best parameter selection

---

## 🏆 Model Evaluation Focus (Medical Context)

Since this is a healthcare application, we emphasize:

- ✅ **Recall (Sensitivity)** – Detect actual heart disease cases  
- ✅ **Specificity** – Identify healthy patients correctly  
- ✅ **Precision** – Reduce false alarms  
- ✅ **F1-Score** – Balanced evaluation  
- ✅ **ROC-AUC** – Overall discriminative power  

---

## 📈 Results

✔ All 4 models implemented and evaluated  
✔ Hyperparameter tuning completed  
✔ Cross-validation applied  
✔ Best performing model selected  

🎯 Target Achieved:  
**ROC-AUC > 0.85**  

---

## 🚀 Deployment Pipeline

### 🔹 Step 1: Model Serialization
- Saved trained model using `pickle`
- Saved preprocessing components
- Stored model metadata

### 🔹 Step 2: FastAPI Backend
- REST API for predictions
- Input validation using Pydantic
- Health check endpoints
- Error handling

### 🔹 Step 3: Docker Containerization
- Dockerfile created
- docker-compose configuration
- Production-ready container setup

---

## 🛠️ Tech Stack

### 📊 Data & ML
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

### ⚙️ Backend
- FastAPI
- Pydantic

### 🐳 Deployment
- Docker
- Docker Compose

---

## 📂 Project Structure

```
Heart-Disease-Detection/
│
├── data/
├── notebooks/
├── models/
├── app/
│   ├── main.py
│   ├── model_loader.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 📌 How to Run the Project

### 1️⃣ Clone the Repository
```
git clone https://github.com/yourusername/heart-disease-detection.git
cd heart-disease-detection
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Run FastAPI
```
uvicorn app.main:app --reload
```

### 4️⃣ Run with Docker
```
docker-compose up --build
```

---

## 💡 Key Learnings

- Practical understanding of classification algorithms  
- Hyperparameter tuning with GridSearchCV  
- Cross-validation in medical datasets  
- Model evaluation in high-stakes domains  
- Building production-ready ML pipelines  

---

## 📌 Future Improvements

- Add SHAP explainability  
- Deploy on cloud (AWS / Azure)  
- Create frontend dashboard  
- Add real-time prediction interface  

---

