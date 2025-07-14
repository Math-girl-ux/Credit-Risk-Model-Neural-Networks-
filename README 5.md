# Credit Risk Modeling on LendingClub Data
### Logistic Regression • XGBoost • Neural Networks

This project uses real-world LendingClub loan data to predict the likelihood of loan default.
We compare three machine learning models — Logistic Regression, XGBoost, and a Neural Network — to evaluate performance on an imbalanced binary classification problem.

---

## Dataset

- **Source**: [LendingClub Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Rows**: 2.2M+
- **Features**: 151 original features (reduced to 14 engineered features)
- **Target Variable**: `loan_status` — mapped to:
  - `0`: Fully Paid
  - `1`: Charged Off (Defaulted)

---

## Models Trained

| Model                | Notes                     |
|---------------------|---------------------------|
| Logistic Regression | Simple, interpretable     |
| XGBoost             | Handles nonlinearity well |
| Neural Network      | Deep learning approach    |

---

## Results Summary

| Model            | Accuracy | ROC-AUC | Recall (Default) | Precision (Default) |
|------------------|----------|---------|------------------|----------------------|
| Logistic Reg.    | 0.806    | 0.706   | 0.05             | 0.54                 |
| XGBoost          | 0.640    | 0.712   | 0.67             | 0.31                 |
| Neural Network   | 0.807    | 0.710   | 0.03             | 0.57                 |

> Due to class imbalance, recall for defaults was low in most models except for XGBoost.

---

## Preprocessing

- Removed irrelevant columns (`id`, `member_id`, etc.)
- Mapped target classes
- Converted categorical variables
- Handled missing values
- Scaled numeric features with `StandardScaler`
