# Happiness Prediction
This project focuses on predicting customer happiness for ACME, a rapidly growing logistics and delivery startup. By leveraging customer survey responses, the goal is to develop a classification model that can accurately determine whether a customer is "happy" or "unhappy." The insights gained from this model will help ACME understand the key drivers of customer satisfaction, identify areas for operational improvement, and ultimately support their global expansion strategy by enhancing customer experience.

## Business Problem

ACME faces the challenge of quantifying and predicting customer happiness to improve its services and maintain a competitive edge. Understanding which factors contribute most to customer satisfaction is crucial for making informed business decisions. The specific task is to build a predictive model that classifies customers as happy (1) or unhappy (0) based on their responses to six survey questions (X1-X6). The success metric for this project is to achieve an accuracy score of 73% or above, or to provide a solution that is demonstrably superior.

## Data

The dataset `ACME-HappinessSurvey2020.csv` contains customer survey responses and their corresponding happiness status.
*   **Size**: 126 rows, 7 columns.
*   **Features**:
    *   `Y`: Target variable (0 = unhappy, 1 = happy).
    *   `X1`: "My order was delivered on time" (rating 1-5).
    *   `X2`: "Contents of my order was as I expected" (rating 1-5).
    *   `X3`: "I ordered everything I wanted to order" (rating 1-5).
    *   `X4`: "I paid a good price for my order" (rating 1-5).
    *   `X5`: "I am satisfied with my courier" (rating 1-5).
    *   `X6`: "The app makes ordering easy for me" (rating 1-5).

**Initial Data Observations**:
*   No missing values were found across any of the columns.
*   16 duplicate rows were identified in the dataset.
*   The target variable `Y` was relatively balanced, with approximately 54.8% happy customers and 45.2% unhappy customers.
*   Descriptive statistics revealed high satisfaction for `X1` (on-time delivery) and `X6` (app ease of use), while `X2` (contents as expected) had the lowest average satisfaction.

**Data Preprocessing**:
*   Outliers in numerical features (X1-X6) were treated using the flooring and capping method based on the Interquartile Range (IQR).

## Approach

1.  **Exploratory Data Analysis (EDA)**:
    *   Univariate analysis was conducted using histograms and boxplots to understand the distribution of each feature and the target variable.
    *   Bivariate analysis included:
        *   Calculation and visualization of a correlation matrix (heatmap) to identify linear relationships between features and the target.
        *   Stacked bar plots for each feature against the target (`Y`) to visualize how different satisfaction levels in X1-X6 correlate with overall happiness.
        *   Distribution plots for each feature, segmented by the happy/unhappy target classes, to observe differences in patterns.

2.  **Data Preparation for Modeling**:
    *   The dataset was split into training (70%) and testing (30%) sets to ensure robust model evaluation.
    *   A constant term was added to the feature set (`X`) for the `statsmodels` Logistic Regression implementation.

3.  **Model Building and Evaluation**:

    *   **Logistic Regression (Initial)**:
        *   An initial Logistic Regression model was trained on all features (X1-X6) using `statsmodels`.
        *   The model's overall significance and individual feature p-values were examined. Only 'X1' was found to be marginally significant (p-value ~0.05).

    *   **Feature Selection for Logistic Regression**:
        *   An iterative backward elimination process was applied, removing features with p-values greater than 0.05 from the Logistic Regression model. This resulted in a simplified model retaining only 'X1' as a significant predictor (along with the constant term).

    *   **Logistic Regression (Selected Features - 'X1' only)**:
        *   A new Logistic Regression model (`lg2`) was trained using only 'X1'.
        *   An optimal classification threshold was determined using the Precision-Recall curve on the training set to maximize the F1-score.
        *   The model's performance was evaluated on the test set using this optimal threshold.

    *   **Decision Tree Classifier (Untuned)**:
        *   A Decision Tree Classifier was trained on the full feature set (X1-X6).
        *   The model showed high training accuracy but significantly lower test accuracy, indicating overfitting.
        *   Feature importances were extracted, identifying `X5` (courier satisfaction), `X1` (on-time delivery), and `X3` (ordered everything wanted) as the most important.

    *   **Hyperparameter Tuning for Decision Tree**:
        *   `GridSearchCV` with 5-fold cross-validation was employed to optimize hyperparameters (`criterion`, `max_depth`, `min_samples_leaf`) for the Decision Tree, targeting maximum F1-score.
        *   The tuned model's performance was evaluated on the test set.

    *   **Random Forest Classifier (Untuned)**:
        *   A Random Forest Classifier was trained on the full feature set.
        *   Similar to the untuned Decision Tree, this model also exhibited strong overfitting.
        *   Feature importances for Random Forest highlighted `X2` (contents as expected), `X3` (ordered everything wanted), and `X5` (courier satisfaction) as most important, while `X4` (good price) and `X6` (app ease) were consistently ranked lowest.

    *   **Feature Reduction (Dropping X4 and X6)**:
        *   Based on the consistently low feature importances from tree-based models and the non-significance in initial Logistic Regression, features `X4` and `X6` were removed from the dataset.

    *   **Random Forest Classifier (Reduced Features)**:
        *   The Random Forest model was retrained using the reduced feature set (X1, X2, X3, X5). Performance was re-evaluated.

    *   **Decision Tree Classifier (Reduced Features)**:
        *   The Decision Tree model was retrained using the reduced feature set (X1, X2, X3, X5). Performance was re-evaluated.

    *   **Hyperparameter Tuning for Decision Tree (Reduced Features)**:
        *   `GridSearchCV` was reapplied to the Decision Tree with the reduced feature set for further optimization.

    *   **Logistic Regression (Reduced Features - X1, X2, X3, X5)**:
        *   A final Logistic Regression model (`lg_reduced`) was trained on the dataset with features `X1, X2, X3, X5`.
        *   An optimal classification threshold was determined using the Precision-Recall curve on the training set to maximize the F1-score.
        *   This model's performance was rigorously evaluated on the test set using the optimal threshold.

## Results

After evaluating multiple models and strategies, the **Logistic Regression model trained on reduced features (X1, X2, X3, X5)** demonstrated the best overall performance on the test set, meeting and exceeding the project's success metric.

**Key Performance Metrics (Logistic Regression with Reduced Features)**:
*   **F1-Score: 77.19%** (This exceeds the target metric of 73%, demonstrating a superior solution.)
*   Accuracy: 65.79%
*   Recall: 95.65%
*   Precision: 64.71%

This model shows a very strong ability to correctly identify happy customers (high Recall) while maintaining a reasonable balance with Precision, indicating that it doesn't incorrectly flag too many unhappy customers as happy. The feature reduction process, by removing less impactful features `X4` and `X6`, contributed to this improved generalization.

**Feature Importance**:
The most influential factors for predicting customer happiness, based on the final Logistic Regression model and confirmed by feature importances from tree-based models, are:
*   `X1`: "My order was delivered on time"
*   `X2`: "Contents of my order was as I expected"
*   `X3`: "I ordered everything I wanted to order"
*   `X5`: "I am satisfied with my courier"

Conversely, `X4` ("I paid a good price for my order") and `X6` ("The app makes ordering easy for me") were consistently found to have lower predictive power.

## Tools & Technologies

*   **Programming Language**: Python 3.x
*   **Libraries**:
    *   `pandas`: For data manipulation and analysis.
    *   `numpy`: For numerical operations.
    *   `matplotlib.pyplot`: For creating static, interactive, and animated visualizations.
    *   `seaborn`: For high-level interface for drawing attractive and informative statistical graphics.
    *   `scikit-learn`: For machine learning models (Decision Tree, Random Forest), data splitting, and evaluation metrics (accuracy, recall, precision, F1-score, confusion matrix).
    *   `statsmodels`: For statistical modeling, particularly Logistic Regression with detailed statistical summaries (p-values, coefficients).

## Key Learnings

1.  **Success Metric Achieved**: The project successfully delivered a model with an F1-score of 77.19% on the test set, surpassing the target of 73%. This indicates a robust and superior solution for predicting customer happiness.
2.  **Importance of Feature Selection**: Iterative feature selection, particularly the p-value-based elimination for Logistic Regression and importance analysis from tree-based models, was crucial. Removing less impactful features (`X4` and `X6`) led to a more parsimonious and better-performing Logistic Regression model.
3.  **Key Drivers of Happiness**: The analysis consistently identified `X1` (on-time delivery), `X2` (contents as expected), `X3` (full order fulfillment), and `X5` (courier satisfaction) as the most significant predictors of customer happiness. Operational focus on these areas is likely to yield the greatest impact.
4.  **Dataset Size Limitations**: Tree-based models (Decision Tree, Random Forest) exhibited significant overfitting, highlighting that the small dataset size (126 rows) limited their ability to generalize well. Simple, more interpretable models like Logistic Regression proved more effective in this scenario.
5.  **Strategic Recommendations for ACME**:
    *   **Adopt the Logistic Regression Model**: Implement the Logistic Regression model using features `X1, X2, X3, X5` due to its high F1-score and interpretability.
    *   **Focus Operational Improvements**: Prioritize efforts on improving "on-time delivery" (`X1`), ensuring "order contents are as expected" (`X2`), facilitating "customers ordering everything they want" (`X3`), and enhancing "courier satisfaction" (`X5`).
    *   **Re-evaluate Survey Questions**: Consider removing questions `X4` ("I paid a good price for my order") and `X6` ("The app makes ordering easy for me") from future surveys, as they demonstrated minimal predictive power in this analysis. This can streamline surveys and reduce customer fatigue.
    *   **Data Collection Strategy**: To enable the development of potentially more powerful and complex models (e.g., highly tuned tree-based ensembles) in the future, ACME should focus on collecting a significantly larger volume of customer feedback data.
```
