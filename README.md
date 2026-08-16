# 2025AC05207_ML_Models_WineQuality
Machine learning models to predict the quality of the Wines.

## a. Problem Statement:
The objective of this assignemnt is to develop, evaluate, and deploy multiple machine learning classification models to predict whether a specific wine is of "Good Quality" (quality score > 5) or "Bad Quality" based on its physicochemical tests. This assignment demonstrates an end-to-end ML deployment workflow, from model training and metric evaluation to serving the model through an interactive Streamlit web application, showing its practical deployment value.

## b. Dataset Description:
  i. Source: UCI Machine Learning Repository (Wine Quality Dataset - combining Red and White wine variants to satisfy the feature constraints).
  ii. Features: 12 independent features (wine_type, fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total   sulfur dioxide, density, pH, sulphates, alcohol).  
  iii. Instances: 6,497 total instances, fulfilling the minimum 500 instance requirement.  
  iv. Target: Binary classification target named quality_label (1 for Good Quality, 0 for Bad Quality).

## c. Github Repository Link:


## d. Models Used:
The following table outlines the evaluation metrics calculated on the test dataset for all five implemented models.
| ML Model Name       |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.7262 | 0.7806 |      0.785  |   0.7998 | 0.7923 | 0.3907 |
| Decision Tree       |     0.7508 | 0.7333 |      0.8213 |   0.7903 | 0.8055 | 0.4596 |
| KNN                 |     0.7469 | 0.8062 |      0.8009 |   0.8151 | 0.8079 | 0.4373 |
| Naive Bayes         |     0.6838 | 0.7465 |      0.762  |   0.7503 | 0.7561 | 0.3071 |
| Random Forest       |     0.8223 | 0.8875 |      0.861  |   0.8681 | 0.8645 | 0.6065 |

Observation on the performance of each model.
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provided a strong baseline model. It effectively captured the linear trends within the chemical data, yielding a respectable AUC of ~0.78 and establishing itself as a solid middle-ground performer. |
| **Decision Tree** | Achieved good accuracy (~0.75) but a noticeably lower AUC (~0.73) compared to Logistic Regression. This indicates that the single tree was prone to overfitting the training data and outputting "hard" probability assignments (mostly 0s and 1s), which naturally lowered the area under the ROC curve. |
| **kNN** | Performed nicely across the board with an accuracy of ~0.75. Implementing StandardScaler in the pipeline was crucial to its success, as it ensured that chemical features with larger measurement scales (like density vs. sulfur dioxide) did not disproportionately skew the distance-based calculations. |
| **Naive Bayes** | Had the lowest performance overall (~0.68 accuracy). Naive Bayes relies on the strict assumption that all features are independent of each other. Because wine chemical properties (e.g., citric acid affecting pH, free sulfur dioxide being a subset of total sulfur dioxide) are highly correlated, this "independence" assumption was violated, causing the metric drop. |
| **Random Forest (Ensemble)** | Delivered outstanding performance, achieving the highest scores across every single metric (Accuracy ~0.82, AUC ~0.89). |
