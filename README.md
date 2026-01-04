# Diabetes Prediction using Logistic Regression

This project uses **Logistic Regression** to predict whether a patient is likely to have diabetes based on medical diagnostic measurements.  
The model is trained on the **Pima Indians Diabetes Dataset**, a well-known dataset for binary classification problems in machine learning.

---

## 📌 Dataset

- **Source:** Pima Indians Diabetes Dataset  
- **URL:** https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv  
- **Target Variable:** `Outcome`
  - `1` → Patient has diabetes  
  - `0` → Patient does not have diabetes  

### Features
| Feature | Description |
|------|------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🚀 Workflow

1. **Import required libraries**
2. **Load dataset from URL**
3. **Explore the dataset**
   - `head()`
   - `info()`
   - `describe()`
4. **Split data into features and target**
5. **Train-Test split**
6. **Train Logistic Regression model**
7. **Evaluate model using:**
   - Accuracy score
   - Confusion matrix
8. **Predict diabetes for new patient data**

---

## 📊 Model Evaluation

- **Accuracy Score** is used to evaluate performance
- **Confusion Matrix** is visualized using a heatmap

The confusion matrix helps identify:
- True Positives
- True Negatives
- False Positives
- False Negatives

---

## 🧪 Sample Prediction

Example input:
```python
input_data = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51]])

## output

Likely to have Diabetes OR

Unlikely to have Diabetes
