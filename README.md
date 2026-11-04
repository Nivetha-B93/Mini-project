# Estimation of Obesity Levels Based on Eating Habits and Physical Condition
A supervised machine learning project that classifies individuals into **seven obesity categories** using their eating habits, lifestyle and physical attributes. The project covers the full data science workflow: EDA, preprocessing, feature selection, model building, evaluation and hyperparameter tuning.

**Author:** Nivetha B | **Batch:** DSB04 | **Project:** Data Science Mini Project

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Key EDA Insights](#key-eda-insights)
- [Models and Results](#models-and-results)
- [Final Model Performance](#final-model-performance)
- [Business Impact](#business-impact)
- [Limitations and Future Work](#limitations-and-future-work)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Author](#author)

---

## Problem Statement

Obesity is a major global health problem linked to diabetes, heart disease and hypertension. In countries like Mexico, Peru and Colombia, changing eating habits and sedentary lifestyles are driving rising obesity rates. Predicting a person's obesity level from their eating habits and physical condition lets healthcare providers and wellness programs intervene early and recommend preventive measures.

## Objectives

- Analyze eating habits and lifestyle patterns across different obesity levels.
- Build predictive models to classify individuals into seven obesity categories.
- Compare multiple machine learning models.
- Identify the lifestyle factors that most influence obesity.
- Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC and a confusion matrix.

**ML problem type:** Supervised Learning, Multi-class Classification

## Dataset

- **Source:** [UCI Machine Learning Repository: Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- **Citation:** Palechor, F. M., & De la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico.* Data in Brief, 25, 104344.
- **Size:** 2,111 rows and 17 columns (16 features + 1 target)
- **Population:** People aged 14 to 61 from Mexico, Peru and Colombia
- **Target variable:** `NObeyesdad` with 7 classes: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, Obesity Type III

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| Age | Numeric | Age in years |
| Height | Numeric | Height in metres |
| Weight | Numeric | Weight in kg |
| family_history_with_overweight | Categorical | Family member with overweight (yes/no) |
| FAVC | Categorical | Frequent consumption of high-calorie food |
| FCVC | Numeric | Frequency of vegetable consumption |
| NCP | Numeric | Number of main meals per day |
| CAEC | Categorical | Eating between meals (no / sometimes / frequently / always) |
| SMOKE | Categorical | Smoker (yes/no) |
| CH2O | Numeric | Daily water consumption |
| SCC | Categorical | Monitors calorie intake (yes/no) |
| FAF | Numeric | Physical activity frequency |
| TUE | Numeric | Time spent using technology devices |
| CALC | Categorical | Alcohol consumption frequency |
| MTRANS | Categorical | Main mode of transportation |

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data manipulation | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Environment | Jupyter Notebook / Google Colab |

## Project Workflow

1. **Problem definition:** objectives and ML problem type.
2. **Exploratory data analysis:** dataset overview, missing values, correlation analysis and 8 visualizations.
3. **Data preprocessing:**
   - Missing values: none found.
   - Duplicates: 24 duplicate rows removed (2,111 to 2,087 records).
   - Outliers: IQR method, with Age, Height and Weight capped (winsorization).
   - Skewness: log transformation applied to Age (skew 0.795 to 0.481).
   - Encoding: ordinal for the target, CAEC and CALC; binary for yes/no and Gender; one-hot for MTRANS.
   - Scaling: StandardScaler fitted on the training set only to avoid data leakage.
   - Split: stratified 80/20 train-test split (1,669 train / 418 test, `random_state=42`).
4. **Feature selection:** correlation analysis, ANOVA F-test and mutual information (SelectKBest). Four weak or redundant features dropped: `SMOKE`, `MTRANS_Automobile`, `MTRANS_Motorbike`, `MTRANS_Bike` (16 final features).
5. **Model building:** six models built with scikit-learn pipelines.
6. **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, 5-fold cross-validation, confusion matrix and an overfitting check.
7. **Hyperparameter tuning:** GridSearchCV with 5-fold stratified cross-validation, scored on weighted F1.

## Key EDA Insights

- **Balanced target:** the seven classes are fairly even (about 267 to 351 records each), so no resampling was needed.
- **Weight is the strongest predictor** (correlation 0.91 with obesity level), followed by family history of overweight (0.51).
- **Family history:** about 46% in Insufficient Weight, rising to nearly 100% from Obesity Type I onward.
- **Age:** most participants are young adults (18 to 26), with a long right tail up to 61.
- **Physical activity (FAF)** declines mildly with obesity level and is lowest in Obesity Type III, though the groups overlap heavily.
- **Middle classes overlap:** Normal Weight and the Overweight classes are the hardest to separate.

## Models and Results

Six models were trained and compared:

1. Logistic Regression (baseline)
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. Support Vector Machine (RBF kernel)
6. Gradient Boosting (`HistGradientBoostingClassifier`)

**Before tuning:** Gradient Boosting ranked first (accuracy 0.9665, F1 0.9664, ROC-AUC 0.9995), followed by Random Forest (accuracy 0.9426). KNN (0.8373) and the Logistic Regression baseline (0.8660) scored lowest.

**After tuning:**

| Model | Accuracy before | Accuracy after |
|---|---|---|
| Gradient Boosting | 0.9665 | **0.9689** |
| Logistic Regression | 0.8660 | 0.9641 |
| KNN | 0.8373 | 0.8445 |

Final ranking after tuning: Gradient Boosting, Logistic Regression, Decision Tree, Random Forest, SVM, KNN. Hyperparameter tuning improved five of the six models, with the biggest gain for Logistic Regression (about +10 F1 points).

## Final Model Performance

**Best model: Gradient Boosting**

| Metric | Score |
|---|---|
| Accuracy | 0.9689 |
| Weighted F1-score | 0.9689 |
| ROC-AUC (OvR, weighted) | 0.9989 |
| Cross-validation F1 (5-fold) | 0.9707 |

- Every class reaches an F1-score of 0.93 or higher, with Obesity Type III (0.99) and Insufficient Weight (0.98) predicted best.
- On the 418 test records the untuned model made only 14 errors, all between neighbouring classes. No record was predicted more than one level away from its true class.
- Overweight Level I was the hardest class, with the lowest recall (0.91).

## Business Impact

- **Early screening:** estimate a person's obesity level from a short lifestyle questionnaire and basic body measurements, and flag high-risk individuals early.
- **Personalized recommendations:** insights into eating habits, activity and family history can guide targeted diet and exercise advice.
- **Cost reduction:** preventing obesity-related conditions lowers long-term healthcare costs.
- **Public health planning:** findings can help prioritize awareness campaigns and interventions.
- **Easy deployment:** the models are lightweight and can be wrapped in a simple web app or API.

## Limitations and Future Work

- Height and Weight closely determine the obesity category (it is defined from BMI), so they dominate predictions. A useful next step is comparing models with and without these two features to measure how much the lifestyle features contribute on their own.
- Most participants are young adults, so the model may generalize less well to older age groups.
- Most remaining errors are between adjacent classes; collecting more data near those class boundaries could help.
- Planned improvements: feature importance / SHAP analysis, XGBoost comparison, and a Streamlit web app for live predictions.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Nivetha-B93/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install the dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Miniproject.ipynb
   ```
4. Run all cells from top to bottom. The dataset loads directly from a URL in the notebook, so an internet connection is needed.

> Note: the Age log transformation cell should be run only once, as running it again applies the log a second time.

## Repository Structure

```
├── Miniproject.ipynb    # Complete analysis and modelling notebook
└── README.md            # Project documentation
```

## Author

**Nivetha B**
Aspiring Data Scientist | Chennai, Tamil Nadu

- LinkedIn: [nivetha-b-822545318](https://www.linkedin.com/in/nivetha-b-822545318)
- GitHub: [Nivetha-B93](https://github.com/Nivetha-B93)
- Email: nivibalan@gmail.com
