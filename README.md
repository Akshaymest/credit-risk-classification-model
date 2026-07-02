# Banking Credit Risk Underwriting & Loan Default Classification

## 🎯 Business Problem
A commercial retail lending institution wants to automate its credit underwriting infrastructure. The core business objective is to accurately flag high-risk loan applications before approval, systematically reducing the bank's exposure to Non-Performing Assets (NPAs) while ensuring creditworthy applicants are processed efficiently.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Environment:** Jupyter Notebook / Anaconda
* **Key Frameworks:** Pandas, NumPy, Scikit-Learn (RandomForestClassifier, train_test_split), Seaborn, Matplotlib.

---

## 📊 End-to-End Machine Learning Pipeline

### 1. Data Ingestion & Preprocessing
* Checked for missing values across 20 distinct applicant columns (0 null items found).
* Handled high-cardinality categorical variables like loan `purpose`, `housing` status, and `credit_history` using **One-Hot Encoding (`pd.get_dummies`)**, expanding the analytical feature space from 20 variables to 48 numeric columns.

### 2. Data Splitting strategy
Divided the 1,000-record dataset into an 80/20 train-test configuration:
* **Training Subsets:** 800 loan profiles to build tree feature-weights.
* **Testing Subsets:** 200 hidden profiles held back to evaluate risk classification behavior on unseen applicants.

---

## 📈 Model Evolution & Optimization Results

### Baseline Model (Unbalanced Random Forest)
The initial baseline model struggled significantly with identifying default conditions due to class imbalances typical in financial data:
* **Overall Accuracy:** 76%
* **Defaulter Recall (Class 'bad'):** 34% (Missed 39 out of 59 actual defaulting accounts)

### Optimized Model (Balanced Hyperparameter Tuning)
Implemented a custom hyperparameter grid specifying entropy-based criterion splits, restricted tree depths to curb over-learning, and active class-weight balancing to penalize missed defaults.

```python
rf_optimized = RandomForestClassifier(
    n_estimators=200, 
    max_depth=6,
    criterion='entropy',
    class_weight='balanced', 
    random_state=42
)```
```

| Model Iteration | Target Accuracy | Defaulter Recall (Catch Rate) | Business Risk (Missed Defaulters) |
| --- | --- | --- | --- |
| **Baseline Model** | 76% | 34% | High Risk (39 Defaults Leaked) |
| **Optimized Model** | **76%** | **69%** | **Minimized Risk (Substantial NPA Reduction)** |

---

## 💡 Strategic Risk Insights

* **The Precision vs. Recall Trade-off:** By deploying the optimized model, the bank successfully catches **double** the volume of risky applicants before lending out capital.
* **Financial Impact:** While catching more defaults increases false alarms slightly (lower precision), the financial savings from avoiding 39 unrecoverable bad loans far outweigh the operational cost of manually reviewing those borderline cases. This model provides an automated, risk-averse gatekeeper for retail credit portfolios.

