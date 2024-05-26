import pandas as pd
import numpy as np
import linear_regression
from sklearn.model_selection import train_test_split
from typing import NoReturn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    X['date'] = pd.to_datetime(X['date'])
    X['year'] = X['date'].dt.year

    # Calculate house age
    X['house_age'] = X['year'] - X['yr_built']
    X['is_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

    #Drop unnecessary or 'wrong' rows
    if X["sqft_living"].isnull().sum() > 0 or X["sqft_lot"].isnull().sum() > 0 or X["sqft_above"].isnull().sum() > 0 or X["sqft_basement"].isnull().sum() > 0 or X["sqft_living15"].isnull().sum() > 0 or X["sqft_lot15"].isnull().sum() > 0:
        X.drop(X.index, inplace=True)
        y = y.drop(y.index)

    # Drop unnecessary columns
    X.drop(columns=['id','date', 'yr_built', 'yr_renovated', 'lat', 'long'], inplace=True)

    # Handle missing values (if any) or Nan values with mean
    X.fillna(X.mean(), inplace=True)

     # Convert categorical features to category type
    X['waterfront'] = X['waterfront'].astype('category')
    X['view'] = X['view'].astype('category')
    X['condition'] = X['condition'].astype('category')
    X['grade'] = X['grade'].astype('category')
    
    # Log transform skewed numeric features
    skewed_features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
    X[skewed_features] = X[skewed_features].apply(lambda x: np.log1p(x))

    # Normalize numeric features
    numeric_features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    return X


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    X['date'] = pd.to_datetime(X['date'])
    X['year'] = X['date'].dt.year

    # Calculate house age
    X['house_age'] = X['year'] - X['yr_built']
    X['is_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

    # Drop unnecessary columns
    X.drop(columns=['id','date', 'yr_built', 'yr_renovated', 'lat', 'long'], inplace=True)

    # Handle missing values (if any) or Nan values with mean
    X.fillna(X.mean(), inplace=True)

     # Convert categorical features to category type
    X['waterfront'] = X['waterfront'].astype('category')
    X['view'] = X['view'].astype('category')
    X['condition'] = X['condition'].astype('category')
    X['grade'] = X['grade'].astype('category')
    
    # Log transform skewed numeric features
    skewed_features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
    X[skewed_features] = X[skewed_features].apply(lambda x: np.log1p(x))

    # Normalize numeric features
    numeric_features = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for column in X.columns:
        x = X[column]

        # Compute Covariance
        covariance = np.mean((x - np.mean(x)) * (y - np.mean(y)))

        # Compute סטיית תקן
        ST_x = np.sqrt(np.mean((x - np.mean(x)) ** 2))
        ST_y = np.sqrt(np.mean((y - np.mean(y)) ** 2))

        # Compute Pearson Correlation Coefficient
        pearson_corr = covariance / (ST_x * ST_y)

        # Plot
        plt.scatter(x, y)
        plt.title(f"{column} - Pearson Correlation: {pearson_corr}")
        plt.xlabel(column)
        plt.ylabel("price")
        plt.savefig(f"{output_path}/{column}.png")
        plt.show()
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=52) #(52 is my lucky number)

    # Question 3 - preprocessing of housing prices train dataset
    X_train_clean = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train_clean, y_train)

    # Question 5 - preprocess the test data
    X_test_clean = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # Initialize variables
    train_size = X_train_clean.shape[0]
    loss = np.zeros((10, 10))
    # for i in range(10):
    #     for j in range(10):
    #         # Sample p% of the overall training data
    #         sample_size = int((i + 1) / 10 * train_size)
    #         sample_indices = np.random.choice(train_size, sample_size, replace=False)
    #         X_sample = X_train_clean.iloc[sample_indices]
    #         y_sample = y_train.iloc[sample_indices]

    #         # Fit linear model (including intercept) over sampled set
    #         #create model that is linear regression class from linear_regression.py
    #         model = linear_regression.LinearRegression(True)
    #         model.fit(X_sample, y_sample)

    #         # Test fitted model over test set
    #         loss[i, j] = model.loss(X_test_clean, y_test)

    #         # Plot
    #         mean_loss = np.mean(loss, axis=1)
    #         std_loss = np.std(loss, axis=1)
    #         plt.plot(np.linspace(10, 100, 10), mean_loss)
    #         plt.fill_between(np.linspace(10, 100, 10), mean_loss - 2 * std_loss, mean_loss + 2 * std_loss, alpha=0.2)
    #         plt.xlabel("Training Size (%)")
    #         plt.ylabel("Loss")
    #         plt.title("Loss as function of Training Size")
    #         plt.show()
    #         plt.close()

            

