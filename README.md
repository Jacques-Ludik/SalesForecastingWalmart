# Walmart Store Sales Forecasting Using Machine Learning

## Overview

This project focuses on forecasting Walmart store sales using machine learning algorithms. We primarily use the RandomForestRegressor from Scikit-learn for this task, leveraging the Walmart Recruiting Store Sales Forecasting dataset available on Kaggle.

## Dataset

The dataset is based on the Walmart Recruiting Store Sales Forecasting competition on Kaggle.

- **Dataset Source**: [Walmart Recruiting Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

## Dependencies

This project makes use of the following libraries:

- Scikit-learn
- Matplotlib
- Seaborn
- OpenDatasets
- Pandas
- NumPy

## Code Structure

1. **Imports**: Required libraries and modules.
2. **Data Loading**: Downloading and unzipping datasets from Kaggle.
3. **Data Manipulation**:
   - Merging datasets.
   - Filling NaN values.
   - Extracting features from date column.
   - Data imputation, scaling, and one-hot encoding.
4. **Data Splitting**: Dividing the dataset into training and validation sets.
5. **Model Training**:
   - Decision Tree Regressor (some parts are commented out).
   - Random Forest Regressor.
6. **Model Evaluation**: Calculating RMSE for training and validation sets.
7. **Hyperparameter Tuning**: Experimenting with different hyperparameters using overfitting curves.
8. **Model Visualization**: Displaying feature importance.
9. **Prediction on Test Data**: Processing and predicting on the test set.

## How to Run

1. Download the dataset from the provided Kaggle link and place it in the same directory as this script.
2. Uncomment sections as needed.
3. Run the Python script to train the models and make predictions.
   ```bash
   python <script_name>.py
   ```

## Results

The results, including RMSE for training and validation sets, as well as feature importances, are printed on the terminal after the script is executed.

## Notes

- Some parts of the code, like decision tree visualizations, feature importances, and some evaluations, are commented out. Uncomment if you wish to execute these sections.
- Ensure you have the required libraries installed. Use pip to install any missing dependencies.
- Adjust the dataset path in the loading functions if you place the dataset in a different directory.

## Contributions

Feel free to fork this project, contribute, and raise any issues or suggestions. We're always looking to improve and collaborate.
