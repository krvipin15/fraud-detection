# 💳 Credit Card Fraud Detection System

![Fraud Detection System](https://cdn.prod.website-files.com/627bf795f0d42b1508d544b1/64d649de97f07ea9e7b7c4b3_CNP-fraud-losses-2018-2023.webp)

## Project Overview

This project addresses the critical challenge of credit card fraud, a significant threat to financial stability. The core difficulty lies in the extreme class imbalance of the dataset, where fraudulent transactions constitute only 0.172% of the data. This imbalance often leads to models that are biased towards the majority class, failing to identify fraud effectively.

The primary objective is to develop and deploy a robust machine learning model that excels at identifying fraudulent activities (high recall) while maintaining a low rate of false positives (high precision) to ensure customer trust and operational efficiency. The entire machine learning lifecycle, from experimentation to deployment readiness, is managed using [MLflow](https://mlflow.org/) to ensure reproducibility and scalability.

## ✨ Features

-   **Data Exploration & Preprocessing:** Comprehensive analysis of credit card transaction data, including handling duplicates, log transformation of skewed features, and robust scaling.
-   **Class Imbalance Handling:** Implementation and evaluation of various resampling techniques: Undersampling (TomekLinks), Oversampling (SMOTE), and Hybrid Sampling (SMOTE + Tomek).
-   **Baseline Model Experimentation:** Training and evaluation of multiple machine learning models (MLPClassifier, Logistic Regression, RandomForest, XGBoost) across different resampled datasets.
-   **MLflow Integration:** Full lifecycle management of machine learning experiments, including tracking of metrics, parameters, artifacts (confusion matrices, ROC curves, PR curves), and model versions.
-   **Hyperparameter Tuning:** Optimization of the best-performing model using [Optuna](https://optuna.org/) to maximize Average Precision.
-   **Model Evaluation & Registration:** Rigorous evaluation of the final model on a hold-out test set and registration to the MLflow Model Registry for seamless deployment.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/fraud-detection.git
    cd fraud-detection
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Download the Dataset:** The `fraud_detection_notebook.ipynb` will automatically download the `creditcardfraud.zip` dataset from Kaggle into the `data/` directory. No manual download is required.

2.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook fraud_detection_notebook.ipynb
    ```
    Open `fraud_detection_notebook.ipynb` in your browser. Run all cells to execute the entire machine learning workflow, from data loading and preprocessing to model training, evaluation, and MLflow tracking.

3.  **View MLflow UI:** To inspect the logged experiments, models, and artifacts, start the MLflow UI:
    ```bash
    mlflow ui
    ```
    Then, open your web browser and navigate to `http://127.0.0.1:5000` (or the address displayed in your terminal).

## 📊 Dataset

The dataset contains credit card transactions made by European cardholders in September 2013. It includes 284,807 transactions, out of which only 492 are fraudulent, making the dataset highly imbalanced (fraud cases represent just 0.172% of the total).

To preserve confidentiality, all features (except `Time` and `Amount`) have been transformed using Principal Component Analysis (PCA), resulting in 28 anonymized features labeled `V1` to `V28`.

-   `Time`: Seconds elapsed since the first transaction in the dataset.
-   `Amount`: Transaction value.
-   `Class`: Target variable, where `1` indicates fraud and `0` indicates a legitimate transaction.

## 🛠️ Technologies Used

-   **Python**
-   **MLflow:** For experiment tracking, model management, and deployment.
-   **Optuna:** For hyperparameter optimization.
-   **Scikit-learn:** For machine learning models (Logistic Regression, RandomForestClassifier, MLPClassifier), data preprocessing (RobustScaler, train_test_split), and evaluation metrics.
-   **Imbalanced-learn:** For handling class imbalance (SMOTE, TomekLinks, SMOTETomek).
-   **XGBoost:** For gradient boosting models.
-   **Pandas & NumPy:** For data manipulation and numerical operations.
-   **Matplotlib & Seaborn:** For data visualization.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Badges

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-orange.svg)](https://mlflow.org/)
