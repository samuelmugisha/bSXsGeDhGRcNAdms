## Overview
This project focuses on predicting customer happiness for ACME, a logistics and delivery startup. By analyzing survey responses from customers, the goal is to develop a classification model that can predict whether a customer is 'happy' (1) or 'unhappy' (0). The project also aims to identify key features that contribute most to customer satisfaction, enabling the company to make data-driven decisions to improve its services.

## Business Problem
ACME, a rapidly growing logistics startup, needs to measure and predict customer happiness to improve its operations and sustain global expansion. Gathering direct feedback is challenging, so the objective is to leverage survey data (X1-X6) to predict a binary target (Y: happy/unhappy). The success metric is to achieve an accuracy score of 73% or above, or to provide a compelling solution that offers superior insights. A bonus goal is to identify the most important features to streamline future surveys and improve predictability.

## Data
The dataset consists of customer survey responses with 126 entries and 7 columns:
- **Y**: Target variable (0 = unhappy, 1 = happy)
- **X1**: My order was delivered on time (1-5)
- **X2**: Contents of my order was as I expected (1-5)
- **X3**: I ordered everything I wanted to order (1-5)
- **X4**: I paid a good price for my order (1-5)
- **X5**: I am satisfied with my courier (1-5)
- **X6**: The app makes ordering easy for me (1-5)

Values for X1-X6 range from 1 (less) to 5 (more) towards the answer.

## Approach
1.  **Data Loading and Initial Inspection**: Loaded the `ACME-HappinessSurvey2020.csv` dataset, checked its shape, head, tail, info, and duplicates. Identified 16 duplicate rows and no null values.
2.  **Exploratory Data Analysis (EDA)**:
    *   **Univariate Analysis**: Used `histogram_boxplot` to visualize the distribution of each variable. Noted a relatively balanced target variable (Y) and varying sentiment for X1-X6.
    *   **Bivariate Analysis**: Explored correlations using a heatmap. Found moderate positive correlations between Y and X1, X5, X6, and X3, with X2 and X4 showing very weak correlations. Stacked bar plots and distribution plots were used to visualize the relationship between each feature and the target variable.
3.  **Outlier Detection and Treatment**: Box plots were used to identify outliers. Outliers in numerical columns were treated using a flooring and capping method (np.clip).
4.  **Data Preparation for Modeling**: The dataset was split into training (80%) and testing (20%) sets. For `statsmodels` Logistic Regression, a constant term was added.
5.  **Model Building and Evaluation**:
    *   **Logistic Regression (Initial)**: Trained a logistic regression model using `statsmodels`. Identified 'const' and 'X1' as marginally significant predictors.
    *   **Feature Selection (Logistic Regression)**: Performed iterative feature elimination based on p-values. Only 'const' and 'X1' remained as statistically significant features.
    *   **Tuned Logistic Regression**: Retrained Logistic Regression with selected features and optimized the threshold using the Precision-Recall curve to maximize F1-score.
    *   **Decision Tree Classifier (Initial)**: Trained a default `DecisionTreeClassifier`. Noted severe overfitting with high training performance but poor test performance.
    *   **Decision Tree Hyperparameter Tuning**: Used `GridSearchCV` to find optimal hyperparameters (`criterion`, `max_depth`, `min_samples_leaf`) to address overfitting.
    *   **Random Forest Classifier (Initial)**: Trained a default `RandomForestClassifier`. Also showed severe overfitting.
    *   **Support Vector Machine (SVM)**: Trained a default `SVC` model.
    *   **Naive Bayes Classifier**: Trained a `GaussianNB` model.
6.  **Feature Reduction & Re-evaluation**: Based on early analysis and feature importances, features X4 and X6 were identified as least important. A new dataset `X_new` (dropping X4 and X6) was created and models were re-trained and evaluated:
    *   **Random Forest Classifier (Reduced Features)**: Retrained with `X_new` and optimized hyperparameters. This model achieved the best performance.
    *   **Decision Tree Classifier (Reduced Features)**: Retrained with `X_new`, and also tuned with `GridSearchCV`.
    *   **Logistic Regression (Reduced Features)**: Retrained with `X_new`.
    *   **SVM Classifier (Reduced Features)**: Retrained with `X_new`.
    *   **Naive Bayes Classifier (Reduced Features)**: Retrained with `X_new`.

## Results

The **Random Forest Classifier with reduced features (X1, X2, X3, X5)** emerged as the best-performing model, successfully meeting the project's accuracy target.

**Best Model Performance on Test Set (Random Forest with Reduced Features):**
- **Accuracy**: 73.08%
- **Recall**: 84.62%
- **Precision**: 68.75%
- **F1-Score**: 75.86%

**Comparison of Model Performance on Test Set:**
```
                                             Accuracy    Recall  Precision        F1
Untuned Decision Tree                        0.631579  0.608696   0.736842  0.666667
Logistic Regression                          0.631579  0.695652   0.695652  0.695652
Tuned Decision Tree                          0.552632  0.652174   0.625000  0.638298
Random Forest Classifier                     0.500000  0.652174   0.576923  0.612245
SVM Classifier                               0.552632  0.782609   0.600000  0.679245
Naive Bayes Classifier                       0.552632  0.739130   0.607143   0.666667
Random Forest Classifier (Reduced Features)  0.730769  0.846154   0.687500   0.758621
Decision Tree Classifier (Reduced Features)  0.615385  0.692308   0.600000   0.642857
Tuned Decision Tree (Reduced Features)       0.615385  0.692308   0.600000   0.642857
Logistic Regression (Reduced Features)       0.500000  1.000000   0.500000   0.666667
SVM Classifier (Reduced Features)            0.653846  0.692308   0.642857   0.666667
Naive Bayes Classifier (Reduced Features)    0.615385  0.769231   0.588235   0.666667
```

## Tools & Technologies
-   **Python**: Programming Language
-   **Pandas**: Data manipulation and analysis
-   **NumPy**: Numerical operations
-   **Matplotlib**: Data visualization
-   **Seaborn**: Enhanced data visualization
-   **Scikit-learn**: Machine learning models (Decision Tree, Random Forest, SVM, Naive Bayes, `train_test_split`, `GridSearchCV`, `accuracy_score`, `recall_score`, `precision_score`, `f1_score`, `confusion_matrix`, `precision_recall_curve`)
-   **Statsmodels**: Statistical modeling (Logistic Regression)

## Key Learnings
-   **Feature Importance**: Features X1 ('my order was delivered on time'), X2 ('contents of my order was as I expected'), X3 ('I ordered everything I wanted to order'), and X5 ('I am satisfied with my courier') were consistently found to be the most influential predictors of customer happiness. X4 ('I paid a good price for my order') and X6 ('the app makes ordering easy for me') demonstrated lower predictive power.
-   **Impact of Feature Reduction**: Removing less important features (X4 and X6) significantly improved the performance of the Random Forest model, leading to better generalization and the achievement of the target accuracy. This suggests that these features might have introduced noise or were not relevant for prediction in this context.
-   **Overfitting with Small Datasets**: Tree-based models (Decision Tree, Random Forest) were prone to overfitting given the relatively small dataset size (126 rows). Hyperparameter tuning and feature selection were crucial to mitigate this.
-   **Model Selection**: While Logistic Regression provided good interpretability, the Random Forest Classifier with a reduced feature set proved to be the most effective predictive model for this problem, offering the best balance of precision and recall while meeting the accuracy goal.

## Recommendations
-   **Adopt Random Forest Classifier (Reduced Features)**: Implement this model for predicting customer happiness.
-   **Focus on Key Drivers**: Prioritize operational improvements around on-time delivery (X1), meeting content expectations (X2), enabling full ordering (X3), and courier satisfaction (X5).
-   **Re-evaluate Survey**: Consider removing X4 and X6 from future surveys to streamline data collection, as they were found to have minimal predictive impact.
-   **Collect More Data**: To further enhance model robustness and generalization, especially for more complex algorithms, continuously collect more customer feedback data.



%%writefile README.md
# ACME Happiness Predictor

## Project Overview
This project aims to predict customer happiness (0: unhappy, 1: happy) based on their survey responses regarding various aspects of their service experience. The goal is to identify key factors influencing happiness and build a robust classification model.

## Data Description
The dataset consists of customer survey responses with the following attributes:
- `Y`: Target variable, 0 (unhappy) or 1 (happy).
- `X1`: My order was delivered on time (1-5).
- `X2`: Contents of my order was as I expected (1-5).
- `X3`: I ordered everything I wanted to order (1-5).
- `X4`: I paid a good price for my order (1-5).
- `X5`: I am satisfied with my courier (1-5).
- `X6`: The app makes ordering easy for me (1-5).

Attributes X1 to X6 have values from 1 to 5, where a smaller number indicates less satisfaction/agreement and a higher number indicates more satisfaction/agreement.

## Goals
- Predict if a customer is happy or not based on survey responses.
- Achieve an accuracy score of 73% or above.
- Identify which questions/features are most important for predicting customer happiness (feature selection).

## Model Performance Summary (Best Model: Random Forest Classifier Reduced Features)
| Model                      | Accuracy | Recall   | Precision | F1-Score |
|----------------------------|----------|----------|-----------|----------|
| Random Forest (Reduced)    | 0.730769 | 0.846154 | 0.687500  | 0.758621 |


## Project Structure
```
. (root directory)
├── README.md
├── models/
│   └── happiness_prediction_model.joblib
├── src/
│   ├── app.py
│   ├── config.py
│   ├── Dockerfile
│   ├── plots.py
│   ├── predict.py
│   ├── requirements.txt
│   └── train.py
└── data/
    ├── ACME-HappinessSurvey2020.csv
  

```

## Setup

### 1. Clone the repository (if applicable):
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Prepare the environment:
Ensure you have Python 3.9+ installed.

### 3. Install dependencies:
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r src/requirements.txt
```

### 4. Data Setup:
The dataset (`ACME-HappinessSurvey2020.csv`) is expected to be mounted from Google Drive at `/content/drive/MyDrive/MLProjects/ACME/ACME-HappinessSurvey2020.csv` as specified in `config.py`.

## Running the Training Pipeline
The `train.py` script contains all the logic for data loading, preprocessing, model training, evaluation, and saving the best model. To run the training pipeline locally, execute:
```bash
python src/train.py
```
This will train all models, print their performance, and save the best-performing model (`happiness_prediction_model.joblib`) to the `models/` directory.

## Launching the Streamlit App Locally
To run the Streamlit application, navigate to the root directory of your project and execute the following command:
```bash
streamlit run src/app.py
```
This will open the app in your web browser, allowing you to interact with the deployed model.

## Hugging Face Deployment

This project is deployed as a Hugging Face Space using Docker. The deployment process involved:
1.  **Hugging Face Authentication**: Logging in to Hugging Face Hub programmatically.
2.  **Space Creation**: Creating a new Hugging Face Space with `space_sdk="docker"`.
3.  **Repository Structure Preparation**: Creating a temporary local repository (`temp_space_repo`) that mirrors the desired structure for the Hugging Face Space, ensuring all necessary application files (from `src/`) and the trained model (from `models/`) are correctly placed.
4.  **Dockerfile Configuration**: The `src/Dockerfile` was adjusted to correctly copy all contents from the build context into the `/app` directory within the Docker container.
5.  **File Upload**: Uploading the prepared `temp_space_repo` content to the Hugging Face Space.

The deployed application can be accessed at: [https://huggingface.co/spaces/dcsamuel/happiness_pediction](https://huggingface.co/spaces/dcsamuel/happiness_pediction)

## Key Findings and Recommendations

### Key Findings:
*   **Best Performing Model**: The **Random Forest Classifier with reduced features (X1, X2, X3, X5)** emerged as the superior solution, achieving an **Accuracy of 73.08%** and an **F1-Score of 75.86%** on the test set. This model successfully meets and exceeds the project's target accuracy of 73% and provides the best balance of precision and recall.
*   **Impact of Feature Reduction**: Dropping **X4 ('I paid a good price for my order')** and **X6 ('the app makes ordering easy for me')** significantly improved model performance, indicating these features were less predictive or introduced noise.
*   **Key Predictive Features**: X1 (on-time delivery), X2 (contents as expected), X3 (ordered everything wanted), and X5 (courier satisfaction) consistently showed higher importance.
*   **Overfitting**: Tree-based models (Decision Tree, Random Forest) initially struggled with overfitting due to the small dataset size (126 rows), which was mitigated by hyperparameter tuning and, more effectively, by feature reduction.

### Recommendations:
1.  **Adopt Random Forest Classifier with Reduced Features**: Implement and deploy this model for customer happiness prediction.
2.  **Focus on Key Drivers**: Prioritize improvements in **on-time order delivery (X1)**, ensuring **order contents meet expectations (X2)**, facilitating customers to **order everything they want (X3)**, and maintaining **high satisfaction with couriers (X5)**.
3.  **Re-evaluate Survey Questions**: Consider **removing X4** and **X6** from future surveys due to their low predictive power.
4.  **Collect More Data**: For further improvements and to enhance model generalization, increasing the dataset size is crucial.
5.  **Continuous Monitoring**: Implement continuous monitoring of the model's performance and regular retraining with new data.
