# Predicting Dementia using Random Forest Model

## Project Overview

This project aims to develop a machine learning model to predict dementia in patients based on MRI imaging and clinical features. Dementia is a progressive neurodegenerative disorder characterized by cognitive decline, impacting memory, reasoning, and daily functioning. Early detection is crucial for timely intervention, particularly in Alzheimer’s disease, one of the most common dementia types.

We use a **Random Forest classifier** to categorize patients into two groups: **Demented** and **Nondemented**. Accurate classification helps support clinical decisions, enabling better patient care and resource management.

---

## Dataset

- **Source:** [OASIS Longitudinal Dataset](https://www.kaggle.com/code/hyunseokc/detecting-early-alzheimer-s/input?select=oasis_longitudinal.csv)
- **Description:** The dataset includes MRI scans and clinical data of 150 subjects aged 60-96.
- **Classes:**  
  - Non-Demented (72 subjects)  
  - Demented (64 subjects)  
  - Converted (14 subjects; initially Non-Demented but later diagnosed with Dementia)
- **Instances:** 373 total
- **Features:** 10 clinical and demographic attributes (Age, Education, SES, MMSE score, etc.)

---

## Project Goals

- Develop a binary classification model to distinguish Demented vs. Nondemented patients.
- Achieve high accuracy, precision, recall, and balanced F1 score to minimize false positives and false negatives.
- Provide an AI tool that can assist in early detection and ongoing monitoring of dementia.

---

## Tech Stack

- Python 3.x  
- Libraries:  
  - `pandas` for data handling  
  - `scikit-learn` for machine learning model, preprocessing, train/test split, evaluation metrics  
  - `matplotlib` / `seaborn` for visualization (confusion matrix)  
- Development environment: Jupyter Notebook / Google Colab / Local IDE

---

## Methodology

1. **Data Loading:** Import and inspect the OASIS dataset.
2. **Preprocessing:**  
   - Encode categorical labels (`Demented` → 0, `Nondemented` → 1).  
   - Drop irrelevant features.  
   - Split data into training (80%) and testing (20%) sets, reserving last two test samples for future testing.
3. **Model Training:**  
   - Initialize Random Forest Classifier with optimized hyperparameters (e.g., 200 trees, entropy criterion).  
   - Train on training data.
4. **Evaluation:**  
   - Use metrics such as accuracy, precision, recall, F1-score, and AUC to evaluate model performance on the test set.  
   - Generate classification and confusion matrix reports.
5. **Visualization:** Present results graphically for interpretability.
6. **Recommendations:** Discuss limitations and suggest improvements like handling class imbalance and threshold tuning.

---

## Evaluation Metrics Explained

- **Accuracy:** Overall correctness of predictions.
- **Precision:** How many predicted positives are actual positives (minimizes false positives).
- **Recall:** How many actual positives were correctly predicted (minimizes false negatives).
- **F1 Score:** Harmonic mean of precision and recall (balances the two).
- **AUC:** Area Under ROC Curve, measures discrimination capability.

---

## Results Summary

- The model achieves strong accuracy and precision, especially in identifying Nondemented patients.
- Recall for Demented class is lower, indicating some missed dementia cases, which is a critical limitation.
- Macro averaging was used for metrics to fairly represent minority classes.

---

## Future Improvements

- **Address Class Imbalance:** Implement oversampling techniques like SMOTE.
- **Feature Engineering:** Add or refine features for better discrimination.
- **Threshold Adjustment:** Tune classification thresholds to improve recall on Demented class.

---

