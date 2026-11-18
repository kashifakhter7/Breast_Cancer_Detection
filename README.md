# 🎗️ Breast Cancer Detection & Classification Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](#)

An end-to-end Machine Learning web application designed to predict whether a breast tumor is **Benign** (non-cancerous) or **Malignant** (cancerous) based on cell nucleus characteristics extracted from digitized Fine Needle Aspirate (FNA) biopsy images.

Built with **Scikit-Learn** and deployed via an interactive **Streamlit** user interface.

---

## 📌 Project Overview

Early detection of breast cancer significantly increases treatment success and patient survival rates. This project provides a diagnostic aid by evaluating 30 key pathological biomarkers and classifying tumor samples into:
- **Benign (Class 1)**: Non-cancerous, healthy tissue.
- **Malignant (Class 0)**: Cancerous, requiring urgent clinical intervention.

The repository includes exploratory data analysis, multi-model benchmarking (evaluating 11 machine learning algorithms), model persistence, and a real-time web interface for clinical parameters.

---

## ✨ Features

- **Multi-Model Benchmark**: Evaluates 11 classification algorithms (Logistic Regression, Random Forest, AdaBoost, SVM, XGBoost, and Ensembles).
- **High-Accuracy Ensemble Classifier**: Best-performing model achieved over **95.6% accuracy** and **97.1% recall**.
- **Interactive Web UI**: Real-time diagnosis via an intuitive Streamlit dashboard.
- **Feature Standardization**: Integrated data scaling (`StandardScaler`) to normalize high-dimensional clinical inputs.
- **Production-Ready**: Serialized model pipeline (`model.pkl`) for instant inference with zero retraining needed.

---

## 🏗️ Project Architecture

```plaintext
Breast_Cancer_Detection/
│
├── ipynb/
│   └── Breast_Cancer_Prediction(updated) (1).ipynb   # Jupyter Notebook with EDA & model training
│
├── pkl_folder/
│   └── model.pkl                                     # Serialized trained AdaBoost model
│
├── python_folder/
│   ├── breast_cancer_detection.py                    # Initial data analysis & Logistic Regression
│   ├── breast_cancer_prediction(updated).py          # Multi-model benchmarking & model export
│   └── frontend.py                                   # Streamlit interactive web dashboard
│
├── requirements.txt                                  # Python dependencies
└── README.md                                         # Project documentation
```

---

## 📊 Dataset Description

The project utilizes the **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset** (from UCI Machine Learning Repository / Scikit-Learn):

- **Total Samples**: 569 instances
- **Target Classes**:
  - `Benign`: 357 instances (62.7%)
  - `Malignant`: 212 instances (37.3%)
- **Number of Features**: 30 numeric, real-valued cell nuclei attributes computed from FNA images:
  - **Mean, Standard Error, and Worst (Largest)** for:
    1. Radius (mean of distances from center to points on the perimeter)
    2. Texture (standard deviation of gray-scale values)
    3. Perimeter
    4. Area
    5. Smoothness (local variation in radius lengths)
    6. Compactness (`perimeter^2 / area - 1.0`)
    7. Concavity (severity of concave portions of the contour)
    8. Concave points (number of concave portions of the contour)
    9. Symmetry
    10. Fractal dimension ("coastline approximation" - 1)

---

## 📈 Model Performance & Comparison

Multiple models were trained and benchmarked on an 80/20 train-test split:

| Algorithm | Accuracy | Precision | Recall | Key Strength |
|:---|:---:|:---:|:---:|:---|
| 🏆 **AdaBoost Classifier** | **95.61%** | **95.71%** | **97.10%** | **Best balanced model; minimizes false negatives** |
| **Extra Trees Classifier** | 95.61% | 95.59% | 94.20% | High variance reduction across trees |
| **Random Forest** | 94.74% | 95.59% | 94.20% | Robust ensemble feature sampling |
| **Voting Classifier** | 93.86% | 94.29% | 95.71% | Blends SVM, Random Forest & Logistic Regression |
| **Naive Bayes (Gaussian)** | 93.86% | 95.59% | 94.20% | Fast probabilistic baseline |
| **XGBoost** | 93.86% | 96.97% | 92.75% | Strong gradient-boosted decision trees |
| **Logistic Regression** | 92.98% | 94.20% | 94.20% | High interpretability |
| **Gradient Boosting** | 92.98% | 94.20% | 94.20% | Sequential error correction |
| **Decision Tree** | 91.23% | 96.83% | 88.41% | Prone to higher variance |
| **K-Nearest Neighbors** | 91.23% | 92.75% | 92.75% | Distance metric sensitive |
| **Support Vector Machine (SVM)** | 90.35% | 90.28% | 94.20% | Margin-based separation |

> **Selection Criteria**: In medical screening, **Recall** is crucial to prevent False Negatives (predicting Benign when cancer is present). **AdaBoost** was selected as the final production model due to its high accuracy (95.61%) and highest recall (97.10%).

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Breast_Cancer_Detection.git
cd Breast_Cancer_Detection
```

### 2. Create a Virtual Environment (Optional but recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Web App
Ensure `model.pkl` from `pkl_folder/` is accessible in your execution directory, then launch:
```bash
streamlit run python_folder/frontend.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 💻 How to Use the Web App

1. Open the application interface in your browser.
2. Enter the 30 cell nucleus values obtained from the biopsy report (or use default values for testing).
3. Click the **Predict** button.
4. The system immediately outputs:
   - 🟢 **Benign: Non-Cancerous** (Low Risk)
   - 🔴 **Malignant: Cancerous** (Immediate medical attention advised)

---

## 🛠️ Technologies Used

- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Model Serialization**: Pickle

---

## 📜 Disclaimer

This application is created for **educational and research purposes only**. It should not be used as an alternative to professional clinical examination, medical advice, or definitive lab diagnosis.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
