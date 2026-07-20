# Titanic Survival Prediction

This notebook demonstrates a machine learning workflow to predict passenger survival on the Titanic, using the well-known Kaggle dataset. The process covers data loading, exploratory data analysis, robust preprocessing with custom scikit-learn transformers and pipelines, model training, hyperparameter tuning, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing Steps](#preprocessing-steps)
- [Model Training](#model-training)
- [Results](#results)
- [Setup and Running the Notebook](#setup-and-running-the-notebook)
- [Key Libraries Used](#key-libraries-used)

## Project Overview

The goal of this project is to build a classification model that predicts whether a passenger survived the Titanic disaster (`Survived` column). The notebook follows best practices for data science, including:

1.  **Data Loading & Initial Exploration**: Importing the dataset and performing initial checks.
2.  **Exploratory Data Analysis (EDA)**: Visualizing correlations within the data.
3.  **Data Splitting**: Using Stratified Shuffle Split to create balanced training and testing sets.
4.  **Feature Engineering & Preprocessing**: Handling missing values, encoding categorical features, and dropping irrelevant columns using a scikit-learn pipeline.
5.  **Feature Scaling**: Standardizing numerical features.
6.  **Model Training**: Training a RandomForestClassifier and tuning hyperparameters using GridSearchCV.
7.  **Model Evaluation**: Assessing the performance of the trained model.
8.  **Prediction**: Generating predictions on the unseen test dataset.

## Dataset
[Kaggle Link to Dataset](https://www.kaggle.com/competitions/titanic)

The dataset used is the classic Titanic dataset, which includes passenger information such as `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.

- `train.csv`: The training data with passenger survival information.
- `test.csv`: The test data for which survival predictions are to be made.

## Preprocessing Steps

To prepare the data for machine learning, a comprehensive pipeline was constructed using custom scikit-learn transformers:

-   **`AgeImputer`**: Fills missing 'Age' values with the mean of the training data.
-   **`FeatureEncoder`**: Converts categorical features ('Sex', 'Embarked') into numerical format using One-Hot Encoding. It also handles potential missing 'Embarked' values.
-   **`FeatureDropper`**: Removes columns deemed irrelevant for prediction, such as 'Name', 'Ticket', and 'Cabin'.
-   **`StandardScaler`**: Scales numerical features to have a mean of 0 and a standard deviation of 1.

## Model Training

A `RandomForestClassifier` was chosen for its robustness and performance. `GridSearchCV` was employed to find the optimal hyperparameters for the model, including `n_estimators`, `max_depth`, and `min_samples_split`.

## Results

The final model achieved an accuracy of approximately **0.82** on the stratified test set.

Predictions for the `test.csv` data are saved to `predictions.csv`, containing `PassengerId` and the predicted `Survived` status.

## Setup and Running the Notebook

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Download Data**: Ensure `train.csv` and `test.csv` are in the root directory (or update paths in the notebook).
3.  **Open in Google Colab**: Upload the `.ipynb` file to Google Colab.
4.  **Run all cells**: Execute all cells in the notebook sequentially.

## Key Libraries Used

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `sklearn` (Scikit-learn) - for preprocessing, model selection, and model training.
-   `tensorflow` (though not explicitly used in the final model training in this notebook)

Feel free to explore and adapt this notebook for further analysis or different models!

## Note

-Some comments are not yet finished, but i'll complete them as soon as possible

- I'm working on web version, i think i'll use streamlit or javascript with API on it, stay tuned!

