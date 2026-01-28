# Email Spam Detection using Ensemble Learning & Explainable AI (SHAP)

## 📌 Project Overview
Email spam continues to be a major challenge, affecting productivity, security, and trust in digital communication. Traditional spam filters often rely on a single machine-learning algorithm, which limits accuracy and adaptability against evolving spam patterns.

This project proposes an **enhanced Email Spam Detection System** using an **Ensemble (Stacking) Machine Learning approach** combined with **Explainable AI (SHAP)** to not only improve prediction accuracy but also explain *why* an email is classified as **Spam** or **Ham**.

---

## 🎯 Objectives
- Accurately classify emails as **Spam** or **Ham**
- Improve performance over single-algorithm models
- Reduce false positives using probability-based decisions
- Provide transparent and interpretable predictions using **SHAP**
- Build an interactive web application for real-time email analysis

---

## 🔍 How This Project Differs from Existing Systems
Most existing spam detection systems (as studied in the base papers) rely on individual classifiers such as **Naïve Bayes**, **SVM**, or **Random Forest**. These approaches suffer from limitations like feature independence assumptions, poor adaptability, or lack of explainability.

### ✅ Key Improvements in Our Approach

| Aspect | Existing Systems | Our System |
|------|----------------|-----------|
| Model Type | Single classifier | **Stacking Ensemble (NB + SVM + RF + GB + MLP)** |
| Accuracy | Moderate | **Higher due to ensemble learning** |
| False Positives | Higher | **Reduced using probability thresholding** |
| Explainability | Not available | **SHAP-based feature explanation** |
| Transparency | Black-box | **Interpretable predictions** |
| UI | Limited / None | **Interactive Streamlit web app** |

---

## 🧠 Algorithms Used
- Naïve Bayes  
- Support Vector Machine (SVM)  
- Random Forest  
- Gradient Boosting  
- Multilayer Perceptron (MLP – Neural Network)  
- **Stacking Classifier (Meta-Learner: Logistic Regression)**  

---

## 🧰 Tools & Libraries
- **Python**
- **Pandas, NumPy** – Data processing
- **Scikit-learn** – Feature extraction and model training
- **SHAP** – Explainable AI (feature contribution analysis)
- **Streamlit** – Web application framework
- **Google Colab** – Model training environment

---

## ▶️ Running the Project

This application **requires pre-trained model files (`.pkl`)** to function.

Due to **GitHub file size limitations**, the trained model files are **not included** in this repository.  
As a result, the Streamlit application **cannot be executed directly after cloning**.

### Required Model Files
To run the application, the following files must be present in the project directory:

- `vectorizer.pkl`
- `selector.pkl`
- `stacked_model.pkl`
- `shap_background.pkl`

These files are generated **after training the model** using the provided dataset.

### How to Obtain the Model Files
- The model is trained using the dataset **`email_spam_preprocess (1).csv`**
- Training is performed in **Google Colab** or a local Python environment
- After training, the required `.pkl` files are generated and placed alongside `app.py`

> **Note:**  
> Without these `.pkl` files, the application interface may load, but prediction and explanation features will not function.

Once the required model files are available, run:
```bash
streamlit run app.py
