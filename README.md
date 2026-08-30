# 🏠 House Price Predictor

An end-to-end **Machine Learning project for predicting residential property prices in Indian cities** using an **XGBoost Regression model** and an interactive **Streamlit web application**.

The project covers the complete workflow from data preprocessing and exploratory analysis to model training, evaluation, model serialization, and deployment through a user-friendly prediction interface.

---

## 🚀 Project Overview

The goal of this project is to estimate the market price of a residential property based on property characteristics such as:

- Location
- BHK
- Carpet area
- Transaction type
- Furnishing
- Facing
- Bathroom count
- Balcony count
- Ownership
- Floor number
- Total floors

The trained XGBoost model is saved as a `.pkl` file and loaded by the Streamlit application to generate price predictions.

---

## 🧠 Machine Learning Workflow

```text
Raw House Price Dataset
        │
        ▼
Data Loading & Exploration
        │
        ▼
Data Cleaning
        │
        ├── Handle missing values
        ├── Remove irrelevant columns
        ├── Convert price to Lakhs
        ├── Extract BHK
        ├── Extract carpet area
        └── Extract floor information
        │
        ▼
Outlier Analysis & Filtering
        │
        ▼
Feature / Target Separation
        │
        ▼
Train-Test Split
        │
        ▼
Feature Preprocessing
        │
        ├── Ordinal Encoding
        └── Robust Scaling
        │
        ▼
Model Training
        │
        ├── Random Forest
        ├── Gradient Boosting
        └── XGBoost
        │
        ▼
Model Evaluation using R²
        │
        ▼
Save Trained Artifacts
        │
        ├── xgb_model.pkl
        └── preprocessor.pkl
        │
        ▼
Streamlit Application
        │
        ▼
House Price Prediction
```

---

## 📊 Dataset

The notebook uses a house-price dataset containing **187,531 records and 21 columns** before preprocessing.

The original dataset includes fields such as:

- `Title`
- `Amount(in rupees)`
- `Price (in rupees)`
- `location`
- `Carpet Area`
- `Status`
- `Floor`
- `Transaction`
- `Furnishing`
- `facing`
- `overlooking`
- `Society`
- `Bathroom`
- `Balcony`
- `Car Parking`
- `Ownership`
- `Super Area`
- `Dimensions`
- `Plot Area`

During preprocessing, irrelevant and highly incomplete fields are removed, and useful property attributes are transformed into model-ready features.

---

## 🧹 Data Preprocessing

The notebook performs several preprocessing steps.

### 1. BHK Extraction

BHK information is extracted from the property title.

Special cases such as:

- Studio
- Apartment
- Builder

are handled before converting BHK into a numeric feature.

### 2. Price Conversion

The original price strings contain values such as:

```text
42 Lac
1.40 Cr
```

These are converted into a common **Lakhs** representation.

For example:

```text
1.40 Cr → 140 Lakhs
42 Lac  → 42 Lakhs
```

### 3. Carpet Area Extraction

The numeric carpet-area value is extracted from strings such as:

```text
500 sqft
779 sqft
```

### 4. Floor Feature Extraction

The `Floor` column is transformed into:

- `floor_number`
- `total_floors`

Special values such as `Ground`, `Upper`, and `Lower` are handled during transformation.

### 5. Missing Values

The project:

- Removes rows where required target values are missing.
- Removes rows with missing critical fields such as transaction and bathroom.
- Uses the **mode** for missing categorical values.
- Uses the **median** for missing numerical values.

### 6. Outlier Filtering

The notebook analyzes numerical outliers and applies domain-based filtering to carpet area and price.

The retained ranges include:

```text
Carpet Area > 200 sqft
Carpet Area < 10,000 sqft

Price > 5 Lakhs
Price < 1,000 Lakhs
```

---

## 🔬 Feature Preparation

The target variable is:

```text
price(in lac)
```

The remaining processed columns are used as model features.

Categorical features are transformed using:

```text
OrdinalEncoder
```

Numerical features are scaled using:

```text
RobustScaler
```

Both transformations are combined using a:

```text
ColumnTransformer
```

The fitted preprocessing object is saved as:

```text
preprocessor.pkl
```

---

## 🤖 Models

Multiple regression models were explored during the project:

### Random Forest Regressor

Used as one of the baseline tree-based regression models.

### Gradient Boosting Regressor

Used as another ensemble regression model for comparison.

### XGBoost Regressor

The final selected model uses:

```text
XGBRegressor(
    learning_rate=0.1,
    max_depth=7,
    n_estimators=600
)
```

These parameters were selected after testing XGBoost hyperparameters using cross-validation / grid-search experimentation in the notebook.

---

## 📈 Model Evaluation

Model performance is evaluated using the:

**R² (R-squared) score**

The notebook evaluates performance on both:

- Training data
- Test data

Five-fold cross-validation is also performed on the training data to estimate model performance across multiple splits.

---

## 💾 Saved Model Artifacts

Two serialized files are generated:

| File | Purpose |
|---|---|
| `xgb_model.pkl` | Trained XGBoost regression model |
| `preprocessor.pkl` | Fitted feature preprocessing pipeline |

The Streamlit application loads these artifacts with `joblib`.

---

## 🖥️ Streamlit Application

The project includes an interactive web application built using **Streamlit**.

The application provides inputs for:

### Location & Deal Type

- City / Area
- Transaction Type
- Status

### Unit Specifications

- BHK
- Bathrooms
- Balconies
- Carpet Area

### Property Attributes

- Furnishing
- Ownership
- Facing
- Total Floors
- Floor Number

After entering the property details, the user clicks:

```text
🔮 Predict Price
```

The application loads the trained model and preprocessing object and returns an estimated market price.

---

## ✨ Application Features

- 🏙️ Multiple Indian city/location options
- 🏠 Property specification inputs
- 🛋️ Furnishing selection
- 📜 Ownership selection
- 🧭 Property facing selection
- 🏢 Floor information
- 🔮 Instant price prediction
- 📋 Displays selected property features
- 💰 Displays estimated market price
- ✅ Input validation for city selection and floor number

The application uses Streamlit session state to retain prediction inputs and results.

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── app.py
├── house_price_prediction.ipynb
├── preprocessor.pkl
├── xgb_model.pkl
├── requirements.txt
└── README.md
```

### File Description

| File | Description |
|---|---|
| `app.py` | Streamlit frontend and prediction logic |
| `house_price_prediction.ipynb` | Data analysis, preprocessing, model experimentation, training, and evaluation |
| `preprocessor.pkl` | Serialized preprocessing pipeline |
| `xgb_model.pkl` | Serialized trained XGBoost model |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd House-Price-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project uses libraries including:

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Matplotlib
- Seaborn

The provided requirements file pins the project's installed package versions, including `numpy`, `pandas`, `scikit-learn`, `streamlit`, and `xgboost`.

---

## ▶️ Run the Application

Make sure the following files are in the same directory as `app.py`:

```text
app.py
preprocessor.pkl
xgb_model.pkl
```

Then run:

```bash
streamlit run app.py
```

Streamlit will launch the application in your browser.

---

## 🔄 Prediction Pipeline

When a user submits a property:

```text
User Input
    ↓
Feature Construction
    ↓
Preprocessor
    ↓
XGBoost Model
    ↓
Predicted Price
    ↓
Streamlit Result
```

The application loads the serialized preprocessing pipeline and XGBoost model using `joblib`, transforms the input, generates the prediction, and displays the estimated price.

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | Scikit-learn |
| Encoding | OrdinalEncoder |
| Scaling | RobustScaler |
| ML Models | Random Forest, Gradient Boosting, XGBoost |
| Final Model | XGBoost Regressor |
| Model Serialization | Joblib |
| Web Application | Streamlit |
| Development Environment | Google Colab / Jupyter Notebook |

---

## 📌 Important Deployment Note

The serialized model and preprocessing pipeline must receive the **same feature schema used during model training**.

If the training notebook or `app.py` is modified, retrain and regenerate:

```text
preprocessor.pkl
xgb_model.pkl
```

before deployment.

This prevents preprocessing/model feature-order or feature-name mismatches during inference.

---

## 🔮 Future Improvements

Potential improvements include:

- Hyperparameter optimization with broader search
- More detailed model evaluation using MAE and RMSE
- Feature importance and explainability
- Better handling of high-cardinality categorical features
- Improved validation of out-of-range user inputs
- Automated model retraining
- Cloud deployment
- Prediction confidence / uncertainty estimates
- Adding a dashboard for property-market analysis

---

## 👨‍💻 Author

**Kesava Pavan**

Machine Learning / Data Science Project

---
