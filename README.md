# Employee Attrition Prediction using Machine Learning

## 📌 Project Overview

Employee attrition is a major challenge for organizations because losing experienced employees can increase recruitment costs, reduce productivity, and affect team performance.

This project uses **Machine Learning** to predict whether an employee is likely to **leave the organization or stay** based on factors such as age, job role, income, job satisfaction, work experience, business travel, distance from home, and other employee-related attributes.

The project was developed as part of **Arya Hackathon Verse 1.0**.

---

## 🎯 Objective

The main objective of this project is to develop a machine learning classification system that can:

* Predict employee attrition.
* Identify employees who may be at risk of leaving.
* Analyze important employee-related factors.
* Compare multiple machine learning classification algorithms.
* Select and save a trained model for future predictions.

---

## 🧠 Problem Statement

Employee turnover can negatively affect an organization's productivity, workforce stability, and operational costs.

The goal of this project is to use historical employee data to build a predictive model that can classify employees into:

* **Yes** → Employee may leave the organization.
* **No** → Employee is likely to stay.

This can help organizations analyze attrition patterns and support data-driven workforce planning.

---

## 📊 Dataset

The project uses `general_data.csv`.

The dataset contains:

* **4,410 employee records**
* **29 columns**

Some important features include:

| Feature                 | Description                               |
| ----------------------- | ----------------------------------------- |
| EmployeeID              | Unique employee identifier                |
| Age                     | Employee age                              |
| Attrition               | Whether the employee left                 |
| BusinessTravel          | Employee travel frequency                 |
| Department              | Employee department                       |
| DistanceFromHome        | Distance from home                        |
| Education               | Education level                           |
| EducationField          | Field of education                        |
| Gender                  | Employee gender                           |
| JobLevel                | Employee job level                        |
| JobRole                 | Employee job role                         |
| MaritalStatus           | Marital status                            |
| MonthlyIncome           | Monthly income                            |
| NumCompaniesWorked      | Number of companies previously worked for |
| PercentSalaryHike       | Percentage salary increase                |
| StockOptionLevel        | Stock option level                        |
| TotalWorkingYears       | Total years of work experience            |
| TrainingTimesLastYear   | Training sessions attended                |
| YearsAtCompany          | Years spent at current company            |
| YearsSinceLastPromotion | Years since last promotion                |
| YearsWithCurrManager    | Years with current manager                |
| EnvironmentSatisfaction | Satisfaction with work environment        |
| JobSatisfaction         | Job satisfaction                          |
| WorkLifeBalance         | Work-life balance                         |
| JobInvolvement          | Level of job involvement                  |
| PerformanceRating       | Employee performance rating               |

The notebook reports 111 missing values and no duplicate rows in the original dataset.

---

## 🔍 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Random Forest Model
   ↓
Model Saving
   ↓
Employee Attrition Prediction
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

### Machine Learning Algorithms

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

---

## 🧹 Data Preprocessing

The project performs several preprocessing steps before model training.

### 1. Missing Value Analysis

Missing values are identified using:

```python
df.isnull().sum().sum()
```

The dataset contains **111 missing values**.

### 2. Duplicate Check

Duplicate records are checked using:

```python
df.duplicated().sum()
```

No duplicate rows were found.

### 3. Removing Unnecessary Features

The following columns are removed before model training:

```text
EmployeeID
EmployeeCount
Over18
StandardHours
```

After removing these columns, the modeling dataset contains **25 columns**.

### 4. Handling Numerical Features

Numerical features are processed using:

* `SimpleImputer`
* `StandardScaler`

### 5. Handling Categorical Features

Categorical features are processed using:

* `SimpleImputer`
* `OneHotEncoder`

The preprocessing operations are integrated into Scikit-learn pipelines.

---

## 🤖 Machine Learning Models

Three classification algorithms are trained and evaluated.

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model.

### 2. Decision Tree

A Decision Tree classifier is trained with:

```python
max_depth=8
```

### 3. Random Forest

The Random Forest classifier uses:

```python
n_estimators=200
random_state=42
class_weight="balanced"
```

The trained Random Forest model is selected as the final model.

---

## 📈 Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   84.81% |    63.33% | 13.38% |   22.09% |
| Decision Tree       |   90.02% |    90.91% | 42.25% |   57.69% |
| Random Forest       |   99.55% |   100.00% | 97.18% |   98.57% |

These values are the recorded results in the project's notebook.

---

## 🏆 Final Model

The project uses **Random Forest Classifier** as the final model.

Recorded test-set performance:

```text
Accuracy  : 99.55%
Precision : 100.00%
Recall    : 97.18%
F1 Score  : 98.57%
```

The notebook then saves the trained pipeline using Joblib.

---

## 💾 Model Saving

The trained model is saved as:

```text
employee_attrition_model.pkl
```

Using:

```python
joblib.dump(
    final_model,
    "employee_attrition_model.pkl"
)
```

This allows the trained model to be reused without retraining it every time.

---

## 🔮 Prediction

The saved model can be used to predict employee attrition.

Example:

```python
prediction = final_model.predict(sample_employee)[0]

if prediction == 1:
    print("Prediction: Employee may leave")
else:
    print("Prediction: Employee likely to stay")
```

The notebook includes a test prediction after saving the final model.

---

## 📁 Project Structure

```text
Arya-Hackathon-Verse-1.0/
│
├── Hackthon.ipynb
├── general_data.csv
├── employee_attrition_model.pkl
├── .ipynb_checkpoints/
└── README.md
```

### File Description

| File                           | Description                                                          |
| ------------------------------ | -------------------------------------------------------------------- |
| `Hackthon.ipynb`               | Complete data analysis, preprocessing, model training and evaluation |
| `general_data.csv`             | Employee dataset                                                     |
| `employee_attrition_model.pkl` | Trained Random Forest model                                          |
| `README.md`                    | Project documentation                                                |

The current GitHub repository contains these project files on the `main` branch.

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/parmveer99solanki99-pixel/Arya-Hackathon-Verse-1.0.git
```

### Step 2: Open the Project

```bash
cd Arya-Hackathon-Verse-1.0
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Open

```text
Hackthon.ipynb
```

### Step 6: Run All Cells

Run the notebook cells sequentially to:

1. Load the dataset
2. Analyze the data
3. Clean the data
4. Preprocess features
5. Train classification models
6. Compare model performance
7. Save the final model
8. Generate predictions

---

## 💡 Key Features

* Employee attrition prediction
* Data cleaning and preprocessing
* Missing value handling
* Categorical feature encoding
* Numerical feature scaling
* Multiple ML model comparison
* Classification metrics
* Random Forest classification
* Model persistence using Joblib
* Employee-level prediction

---

## 📌 Applications

This project can be used as a prototype for:

* Employee retention analysis
* HR analytics
* Workforce planning
* Attrition risk identification
* Data-driven HR decision support
* Employee engagement analysis

---

## 🔮 Future Improvements

Future versions of the project can include:

* Interactive Streamlit dashboard
* Employee attrition probability score
* Feature importance visualization
* SHAP-based model explainability
* Hyperparameter tuning
* Cross-validation
* ROC-AUC analysis
* Real-time employee prediction interface
* Automated HR analytics dashboard
* Cloud deployment
* REST API for model predictions

---

## ⚠️ Important Note

The reported model performance comes from the current notebook's train-test evaluation. Before using such a model in a real HR decision-making system, additional validation, leakage checks, fairness analysis, calibration, and evaluation on genuinely unseen organizational data would be necessary.

## 👨‍💻 Author

**Paramveer Singh Solanki**

B-Tech Computer Science and Engineering
Arya College of Engineering, Jaipur

GitHub:
https://github.com/parmveer99solanki99-pixel

---

## ⭐ Acknowledgement

This project was developed as part of **Arya Hackathon Verse 1.0** with the objective of applying Machine Learning techniques to a real-world employee analytics problem.

---

## 📜 License

This project is intended for educational and hackathon purposes.
