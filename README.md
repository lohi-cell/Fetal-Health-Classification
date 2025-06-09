# Fetal Health Classification

## Project Overview
This project aims to develop machine learning models to classify fetal health outcomes using Cardiotocography (CTG) data. CTG is a non-invasive and cost-effective diagnostic tool used during pregnancy to monitor fetal heart rate (FHR) and uterine contractions. The goal is to predict whether the fetus is in a Normal, Suspect, or Pathological state to help detect fetal distress early and support timely medical intervention.

## Dataset
- Source: [Kaggle - Fetal Health Classification](https://www.kaggle.com/code/karnikakapoor/fetal-health-classification)
- Records: 2126 labeled instances
- Classes:  
  - 1: Normal  
  - 2: Suspect  
  - 3: Pathological
- Features include fetal heart rate variability, uterine contractions, fetal movements, and histogram measures.

## Project Objectives
- Build and evaluate machine learning models to classify fetal health using CTG data.
- Enable early detection of fetal distress to improve maternal and fetal health outcomes.
- Assist healthcare professionals in assessing pregnancy risks.

## Technologies & Tools
- Python (Google Colab environment)
- Libraries:
  - NumPy (Numerical computations)
  - Pandas (Data manipulation)
  - Scikit-learn (Machine learning models)
  - Matplotlib & Seaborn (Data visualization)

## Methodology
1. **Data Preprocessing**
   - Handle missing data through imputation.
   - Normalize and scale feature values.
   - Feature selection and extraction.

2. **Model Building**
   - Train models including:
     - Linear Regression
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier
     - K-Means Clustering

3. **Model Evaluation**
   - Assess models with metrics like Accuracy, Precision, Recall, F1-Score.
   - Compare training and testing performance.
   - Fine-tune hyperparameters to improve accuracy.

4. **Prediction**
   - Use the best-performing model to classify fetal health as Normal, Suspect, or Pathological.

## How to Run
1. Clone the repository or download the project files.
2. Open the Jupyter notebook (`.ipynb`) in Google Colab.
3. Upload the dataset file `fetal_health.csv` or download it directly within the notebook.
4. Run all cells step-by-step for data preprocessing, model training, evaluation, and prediction.

## Results
- The model provides reliable classification of fetal health status.
- Enables healthcare professionals to make timely decisions for prenatal care.
