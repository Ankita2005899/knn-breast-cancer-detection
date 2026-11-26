# Breast Cancer Detection using KNN

Classifies breast tumors as **malignant or benign** from 30 cell-nucleus measurements (Breast Cancer Wisconsin Diagnostic dataset, 569 samples).

## Workflow
EDA → stratified train/test split → scaling comparison → best K via 10-fold CV → GridSearchCV (K, metric, weights, p) in a Pipeline (no data leakage) → evaluation → PCA decision-boundary plot → comparison with other models → saved model + new-patient prediction

## Key results (114 unseen test samples)
| Metric | Score |
|---|---|
| Accuracy | 95.61% |
| Precision | 93.51% |
| F1-score | 96.64% |
| ROC-AUC | 99.26% |
| Malignant recall | 88.1% (5 of 42 malignant missed, 0 false alarms) |

- Feature scaling raised accuracy from **91.23% to 95.61%**.
- Best parameters: Euclidean distance, K=23, uniform weights.

## Model comparison
| Model | Accuracy |
|---|---|
| Logistic Regression | 98.25% |
| SVM | 98.25% |
| KNN (tuned) | 95.61% |
| Random Forest | 95.61% |
| Decision Tree | 91.23% |

## Limitations
The tuned KNN missed 5 malignant tumors, and Logistic Regression and SVM performed better on this dataset. Future work: optimise directly for malignant recall and adjust the decision threshold.

## Run it
pip install -r requirements.txt
Open `KNN_Breast_Cancer_Detection.ipynb` in Jupyter or Google Colab and run all cells.

## Disclaimer
Educational project only. Not for medical diagnosis.
