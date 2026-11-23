# 💳 Credit Risk Prediction using ANN

**An End-to-End Deep Learning Project with PyTorch, FastAPI & Docker**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## 📌 Overview

Credit risk prediction helps financial institutions identify borrowers who are likely to default on their loans. Predicting credit risk enables lenders to make more informed lending decisions and manage potential financial losses.

This project develops an end-to-end Deep Learning solution using an Artificial Neural Network (ANN) built with PyTorch to predict whether a loan applicant is likely to default. It covers the complete workflow, from data preprocessing and model training to evaluation and deployment using FastAPI and Docker.

## 🎯 Project Objectives

- Perform Exploratory Data Analysis (EDA) to understand borrower characteristics.
- Clean and preprocess financial and loan-related data.
- Prepare numerical and categorical features for deep learning.
- Build an Artificial Neural Network using PyTorch.
- Train the ANN for binary classification.
- Evaluate the model using classification metrics.
- Save the trained model and preprocessing components.
- Deploy the model using FastAPI.
- Containerize the API using Docker.

## ⚙️ Deep Learning Workflow

```text
Data Collection
      |
Data Cleaning & Preprocessing
      |
Exploratory Data Analysis (EDA)
      |
Feature Transformation
      |
Train-Test Split
      |
Tensor Conversion & DataLoader
      |
Artificial Neural Network (ANN)
      |
Model Training
      |
Model Evaluation
      |
Model Serialization (Joblib)
      |
FastAPI Application
      |
Docker Containerization
      |
Credit Risk Prediction API
```

## 📈 Model Performance

### Performance Metrics

| **Metric** | **Score** |
|---|---|
| Accuracy | 91% |
| Precision | 82% |
| Recall | 75% |
| F1 Score | 78% |
| ROC-AUC | 91% |

### Classification Report

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|---|---:|---:|---:|---:|
| 0 (Non-Default) | 0.93 | 0.96 | 0.94 | 5,066 |
| 1 (Default) | 0.82 | 0.75 | 0.78 | 1,418 |
| Accuracy | | | 0.91 | 6,484 |
| Macro Avg | 0.88 | 0.85 | 0.86 | 6,484 |
| Weighted Avg | 0.91 | 0.91 | 0.91 | 6,484 |


### 📊 Model Evaluation Highlights

- Achieved 91% accuracy on the unseen test dataset.
- Achieved 82% precision, indicating strong reliability among predicted loan defaulters.
- Successfully identified approximately 75% of actual loan defaulters.
- Achieved a 91% ROC-AUC score, demonstrating strong classification performance.
- Attained a 78% F1-score, maintaining a good balance between precision and recall for credit risk prediction.

## 🔍 Key Findings

- The model uses applicant financial, employment, and loan-related information to predict credit risk.
- Neural networks can learn non-linear relationships between borrower features.
- Default prediction is a binary classification problem.
- Evaluation of the default class is important because the dataset may contain class imbalance.

## 🚀 Model Deployment

The trained ANN and preprocessing components are saved and integrated into a FastAPI application.

The application exposes a REST API that accepts loan applicant information and returns a credit risk prediction.

## 🛠️ Technologies Used

| **Category** | **Technologies** |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| API Development | FastAPI |
| Model Serialization | Joblib |
| Deployment | Docker |
| Tools | Jupyter Notebook, VS Code |

## 📁 Project Structure

```text
Credit-risk-prediction-ANN/
│
├── app/
│   └── main.py
│
├── data/
│
├── notebooks/
│
├── Dockerfile
│
├── credit_risk_model.pkl
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 💻 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/tirth-dabhi16/Credit-risk-prediction-ANN.git
```

### 2. Navigate to the Project

```bash
cd Credit-risk-prediction-ANN
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Application

```bash
uvicorn app.main:app --reload
```

### 5. Access API Documentation

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

Use the interactive Swagger UI to test the credit risk prediction API.

### Run using Docker Hub

Pull the image:

```bash
docker pull tirthdabhi/credit-risk-ann-api
```

Run the container:

```bash
docker run -d -p 8000:8000 --name credit-risk-api tirthdabhi/credit-risk-ann-api
```

Access the API:

```text
http://localhost:8000/docs
```
### Run with Docker Compose

Start the application:

```bash
docker compose up -d
```

Stop the application:

```bash
docker compose down
```

## 📚 What I Learned

- Building Artificial Neural Networks using PyTorch.
- Understanding neural network architecture and forward propagation.
- Working with structured financial datasets.
- Preprocessing numerical and categorical features.
- Training neural networks for binary classification.
- Evaluating models using classification metrics.
- Integrating deep learning models with FastAPI.
- Deploying ML APIs using Docker.
- Running containerized applications across environments.

## 🙏 Acknowledgement

This project helped me strengthen my practical understanding of Deep Learning and model deployment. I gained hands-on experience with PyTorch, neural network training, preprocessing, model evaluation, FastAPI, and Docker while building an end-to-end credit risk prediction system.

Thank you for visiting this repository! ⭐

## 👨‍💻 Author

**Tirth Dabhi**

Computer Engineering Student | Data Science & AI Enthusiast
