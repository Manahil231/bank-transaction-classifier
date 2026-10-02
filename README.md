# Bank Transaction Predictor

A end-to-end Machine Learning project. It takes the details of a bank transaction and predicts whether it is a **Withdrawal** or a **Deposit**. The final model is served through a **Streamlit** web app.

**Live app:** LIVE_APP_LINK_HERE

---

## 1. Problem

Given a transaction's amount, date, account and branch, can we predict if it is a **Withdrawal** or a **Deposit**?

This is a **binary classification** problem.

- Target: `Transaction_Type` (Deposit = 0, Withdrawal = 1)
- In the app: **YES** = Withdrawal predicted, **NO** = Deposit

## 2. Dataset

| Item | Detail |
|---|---|
| Source | 91 daily transaction CSV files (`data/daily/`) + customer list (`data/accounts.csv`) |
| Rows | 878 transactions |
| Accounts | 10 accounts (A001 - A010) |
| Branches | Karachi, Lahore, Islamabad |
| Account types | Saving, Current |
| Cleaned file | `bank_transaction_data.csv` |

Main columns: `Date`, `Account_ID`, `Transaction_Type`, `Amount`, `Branch`, `Account_Type`.

## 3. Approach

1. **Load data:** joined all 91 daily files into one table, created a unique `Transaction_ID` for each row, and merged `Account_Type` from `accounts.csv`.
2. **Data quality checks:** missing values (none), duplicates, datatypes, spread, skewness (Amount about 0.43) and outliers by the IQR method (none).
3. **Cleaning:** converted `Date` to datetime and created `Month`, `Day`, `DayOfWeek`. One pair of identical-looking rows had different Transaction IDs, so both were kept as genuine separate transactions.
4. **EDA:** target balance, Amount distribution, Amount by transaction type, branch and account type, weekday and correlation heatmap.
5. **Features:**
   - Numeric: `Amount`, `Month`, `Day`, `DayOfWeek`
   - Categorical: `Account_ID`, `Branch`, `Account_Type`
6. **Train-test split:** 80% / 20%, `random_state=42`, stratified.
7. **Preprocessing (in a Pipeline):**
   - Numeric: median imputer + `StandardScaler`
   - Categorical: most-frequent imputer + `OneHotEncoder`
8. **Models compared:** Dummy baseline, Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting.
9. **Model selection:** by 5-fold cross-validated ROC-AUC on the training data only (the test set was not used to choose the model).
10. **Saved** the full preprocessing + model pipeline as `bank_transaction.pkl`, then reloaded it and made a manual prediction to confirm it works.

## 4. Results

Test-set scores (from `model_training.ipynb`):

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **Decision Tree (final)** | 0.680 | 0.670 | 0.588 | 0.976 | 0.734 | 0.725 |
| Gradient Boosting | 0.678 | 0.665 | 0.629 | 0.683 | 0.655 | 0.750 |
| Random Forest | 0.664 | 0.653 | 0.604 | 0.744 | 0.667 | 0.741 |
| Logistic Regression | 0.653 | 0.648 | 0.635 | 0.573 | 0.603 | 0.715 |
| KNN | 0.604 | 0.602 | 0.575 | 0.561 | 0.568 | 0.652 |
| Dummy (baseline) | 0.500 | 0.534 | 0.000 | 0.000 | 0.000 | 0.500 |

**Final model: Decision Tree.** It had the highest cross-validated ROC-AUC, so it was chosen by that rule.


## 5. How to run the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Example predictions tested in the app

All four examples use the transaction date 2 October 2026.

| Amount | Account ID | Branch | Account Type | Prediction (Withdrawal?) | Withdrawal chance |
|---|---|---|---|---|---|
| 2,000 | A001 | Karachi | Saving | YES | 100% |
| 10,000 | A004 | Karachi | Saving | YES | 56% |
| 40,000 | A005 | Lahore | Current | YES | 56% |
| 75,000 | A009 | Islamabad | Saving | NO | 0% |

Small amounts are predicted as Withdrawals and large amounts as Deposits, which matches the pattern seen in the data.

## 6. Project structure

```
bank_transaction/
├── app.py                      # Streamlit app
├── model_training.ipynb        # full ML workflow
├── bank_transaction.pkl        # saved preprocessing + model pipeline
├── bank_transaction_data.csv   # cleaned dataset
├── requirements.txt            # exact library versions
├── README.md
└── data/
    ├── accounts.csv
    └── daily/                  # 91 raw daily files
```

## 7. Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, Streamlit.

## 8. Limitations and future work

- Small dataset (878 rows) and few useful features.
- Adding more information (for example, a transaction description or account balance) would likely improve accuracy a lot.
- Try hyperparameter tuning and threshold tuning to balance precision and recall.
