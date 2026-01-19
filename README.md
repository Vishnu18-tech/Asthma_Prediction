# Asthma Prediction using Machine Learning

This project aims to predict whether a patient has asthma using clinical, lifestyle, and environmental features.

---

## 📊 Dataset Overview
- Total records: 10,000
- Combination of numerical and categorical features
- Target variable: **Has_Asthma** (0 = No, 1 = Yes)
- The dataset is imbalanced, with fewer asthma-positive cases

---

## 🤖 Models Implemented
- **Logistic Regression**
- **Support Vector Machine (SVM)**

Both models were trained using the same preprocessing pipeline for fair comparison.

---

## 🖼️ Model Performance Screenshots

### Logistic Regression Metrics
![Logistic Regression Metrics](assets/Screenshot%202026-01-18%20234034.png)

### Support Vector Machine (SVM) Metrics
![SVM Metrics]([assets/svm_metrics.png](https://github.com/Vishnu18-tech/Asthma_Prediction/blob/main/assets/Screenshot%202026-01-18%20234252.png))

---

## Observation
**Logistic Regression and SVM produced identical results, indicating that the dataset is close to linearly separable after preprocessing and class balancing.**
