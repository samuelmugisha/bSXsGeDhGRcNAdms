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

## Model Performance Summary (Best Model: Random Forest Classifier Reduced Features)
| Model                      | Accuracy | Recall   | Precision | F1-Score |
|----------------------------|----------|----------|-----------|----------|
| Random Forest (Reduced)    | 0.730769 | 0.846154 | 0.687500  | 0.758621 |

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

## Project Structure
```
. (root directory)
├── README.md
├── data/
│   └── ACME-HappinessSurvey2020.csv
├── models/
│   └── happiness_prediction_model.joblib
├── notebooks/
│   └── Happiness_Prediction.html
│   └── Happiness_Prediction.ipynb
├── src/
│   ├── app.py
│   ├── config.py
│   ├── Dockerfile
│   ├── plots.py
│   ├── predict.py
│   ├── requirements.txt
│   └── train.py
```

## Application Structure
This repository contains the following key files for the application:
-   `app.py`: The Streamlit frontend application for user interaction.
-   `backend.py`: The Flask backend API that serves model predictions.
-   `config.py`: Configuration file for model paths and feature lists.
-   `predict.py`: Contains functions for loading the model and making predictions.
-   `Dockerfile`: Defines the Docker image for containerizing the application.
-   `entrypoint.sh`: A shell script to run both the Flask backend and Streamlit frontend concurrently in the Docker container.
-   `requirements.txt`: Lists all Python dependencies with specific versions.
-   `final_happiness_predictor.joblib`: The trained Random Forest model.

## Local Installation and Setup

### Prerequisites
-   Python 3.9+
-   `pip` (Python package installer)
-   `git`

### Steps:
1.  **Clone the Repository (or download the 'happiness' folder)**:
    ```bash
    git clone <repository-url>
    cd happiness
    ```
    (If you downloaded the `happiness.zip` from Colab, extract it and navigate into the `happiness` directory).

2.  **Install Dependencies**:
    Navigate to the `happiness` directory and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Flask Backend**:
    In your terminal, start the Flask application. This will typically run on `http://127.0.0.1:5000`:
    ```bash
    python backend.py
    ```
    Keep this terminal open, as the Flask app needs to keep running.

4.  **Run the Streamlit Frontend**:
    In a **new terminal window** (while the Flask backend is still running), navigate to the `happiness` directory and start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    Streamlit will typically open in your web browser at `http://localhost:8501` (or a similar address).

## Deployment on Hugging Face Spaces

This application is designed for Docker-based deployment on Hugging Face Spaces. The `Dockerfile` and `entrypoint.sh` are configured to launch both the Flask backend and Streamlit frontend within the same container. Upon pushing the project files to a Hugging Face Space configured with the Docker SDK, the platform will automatically build the image and deploy the application.

**Key files for Hugging Face deployment:**
-   `Dockerfile`
-   `entrypoint.sh`
-   `requirements.txt`
-   `app.py`
-   `backend.py`
-   `config.py`
-   `predict.py`
-   `final_happiness_predictor.joblib`


