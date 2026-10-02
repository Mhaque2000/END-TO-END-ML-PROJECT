# 📊 Student Exam Performance Prediction

An end-to-end Machine Learning project that predicts a student's **Mathematics score** based on demographic information, parental education, lunch type, test preparation, reading score, and writing score.

The project follows a modular ML pipeline architecture covering **data ingestion, data transformation, model training, model evaluation, serialization, and prediction through a Flask web application**.

---

## 🚀 Project Overview

Student academic performance can be influenced by several demographic and educational factors.

This project uses the **Student Performance Dataset** to build a regression model that predicts a student's `math_score`.

### 🎯 Objective

Build an end-to-end regression system that:

* Loads and validates the dataset
* Performs data preprocessing
* Handles categorical and numerical features
* Trains multiple regression models
* Compares model performance
* Selects the best-performing model
* Saves the trained model and preprocessor
* Provides a web interface for making predictions

---

## 📂 Dataset

The project uses the **Student Performance Dataset** containing **1,000 student records**.

### Target Variable

* `math_score`

### Features

#### Categorical Features

* `gender`
* `race_ethnicity`
* `parental_level_of_education`
* `lunch`
* `test_preparation_course`

#### Numerical Features

* `reading_score`
* `writing_score`

### Dataset Split

| Dataset  | Records |
| -------- | ------: |
| Total    |   1,000 |
| Training |     800 |
| Testing  |     200 |

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Data Science & Machine Learning

* NumPy
* Pandas
* Scikit-learn
* XGBoost
* CatBoost

### Preprocessing

* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`
* `ColumnTransformer`
* Scikit-learn Pipelines

### Web Application

* Flask
* Jinja2
* Bootstrap

### Model Serialization

* Dill

### Development

* Logging
* Custom Exception Handling
* Modular Python Package Structure

---

## 🏗️ Project Architecture

```text
end-to-end-ml-project/
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── logs/
│
├── notebook/
│   └── data_analysis.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── app.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## 🔄 Machine Learning Workflow

```text
                 ┌──────────────────┐
                 │ Student Dataset  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Data Ingestion   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Data Validation  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────────┐
                 │ Data Transformation  │
                 │                      │
                 │ Imputation           │
                 │ Scaling              │
                 │ One-Hot Encoding     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Model Training       │
                 │                      │
                 │ Linear Regression    │
                 │ Decision Tree        │
                 │ Random Forest        │
                 │ Gradient Boosting    │
                 │ AdaBoost             │
                 │ KNN                  │
                 │ XGBoost              │
                 │ CatBoost             │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Model Evaluation     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Best Model           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ model.pkl            │
                 │ preprocessor.pkl     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Flask Web Application │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Math Score Prediction│
                 └──────────────────────┘
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline handles both numerical and categorical data.

### Numerical Pipeline

```text
Missing Values
      ↓
SimpleImputer
      ↓
StandardScaler
```

### Categorical Pipeline

```text
Missing Values
      ↓
SimpleImputer
      ↓
OneHotEncoder
```

Both pipelines are combined using `ColumnTransformer`.

The transformed training data contains **19 features** after categorical encoding.

---

## 🤖 Machine Learning Models

The following regression algorithms are evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. AdaBoost Regressor
6. K-Nearest Neighbors Regressor
7. XGBoost Regressor
8. CatBoost Regressor

Hyperparameter tuning is performed using **GridSearchCV with 5-fold cross-validation**.

### Evaluation Metric

The primary evaluation metric used during model comparison is:

**R² Score**

Additional regression metrics such as **MAE** and **RMSE** can also be used to provide a broader view of prediction error.

---

## 📈 Model Evaluation

The project compares the performance of the different regression algorithms and selects the best-performing model based on the evaluation process.

> **Note:** Add your actual model scores below after the final training run.

| Model             | Train R² | Test R² |
| ----------------- | -------: | ------: |
| Linear Regression |        — |       — |
| Decision Tree     |        — |       — |
| Random Forest     |        — |       — |
| Gradient Boosting |        — |       — |
| AdaBoost          |        — |       — |
| KNN               |        — |       — |
| XGBoost           |        — |       — |
| CatBoost          |        — |       — |

The selected model is saved as:

```text
artifacts/model.pkl
```

The fitted preprocessing pipeline is saved as:

```text
artifacts/preprocessor.pkl
```

---

## 🌐 Web Application

The trained model is integrated with a **Flask web application**.

Users can enter student information through a web form and receive a predicted mathematics score.

### Prediction Inputs

* Gender
* Race/Ethnicity
* Parental Level of Education
* Lunch
* Test Preparation Course
* Reading Score
* Writing Score

### Prediction Flow

```text
User Input
    ↓
Flask Application
    ↓
Prediction Pipeline
    ↓
Saved Preprocessor
    ↓
Saved ML Model
    ↓
Predicted Math Score
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd end-to-end-ml-project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

---

## 🧪 Training the Model

The training pipeline can be executed to perform:

```text
Data Ingestion
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Model Serialization
```

The resulting model and preprocessing objects are stored in the `artifacts/` directory.

---

## 📦 Generated Artifacts

| File               | Purpose                        |
| ------------------ | ------------------------------ |
| `model.pkl`        | Trained machine learning model |
| `preprocessor.pkl` | Fitted preprocessing pipeline  |

---

## 🧰 Error Handling & Logging

The project includes:

* Custom exception handling
* Centralized logging
* Modular components
* Reusable prediction pipeline
* Separate training and prediction workflows

This makes the project easier to debug, maintain, and extend.

---

## 🔮 Future Improvements

* Add MAE and RMSE comparison to the model evaluation
* Add automated model evaluation reports
* Add unit and integration tests
* Containerize the application with Docker
* Deploy the Flask application to a cloud platform
* Add CI/CD using GitHub Actions
* Add model monitoring
* Add API endpoints for programmatic predictions

---

## 📌 Key Learning Outcomes

Through this project, the following concepts were implemented:

* End-to-end ML project development
* Data ingestion
* Data preprocessing
* Feature engineering
* Categorical feature encoding
* Numerical feature scaling
* Scikit-learn pipelines
* Multiple regression algorithms
* Hyperparameter tuning
* Cross-validation
* Model evaluation
* Model serialization
* Flask deployment
* Logging and exception handling
* Modular project architecture

---

## 👨‍💻 Author

**Mahamood**

This project was built as part of my journey toward developing practical **Machine Learning and AI Engineering** skills.

---

## ⭐ If You Find This Project Useful

If you find this project helpful, consider giving the repository a ⭐ on GitHub.
