import pandas as pd
import numpy as np
from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split
from typing import NoReturn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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

    X['house_age'] = X['year'] - X['yr_built'] # Calculate house age and add as new feature
    X['is_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # Add new feature to indicate if house was renovated

    #Drop unnecessary or 'wrong' rows
    for row in X.iterrows():
        if row[1]['sqft_living'] <= 0 or row[1]['sqft_lot'] <= 0 or row[1]['sqft_above'] <= 0 or row[1]['sqft_basement'] < 0 or row[1]['sqft_living15'] <= 0 or row[1]['sqft_lot15'] <= 0:
            X.drop(index=row[0], inplace=True)
            y.drop(index=row[0], inplace=True)
    
    y.fillna(y.mean(), inplace=True) # Handle missing values (if any) or Nan values with mean

    # Drop unnecessary columns entirely
    X.drop(columns=['id','date', 'yr_built', 'yr_renovated', 'lat', 'long', 'year'], inplace=True)

    X.fillna(X.mean(), inplace=True) # Handle missing values (if any) or Nan values with mean
    
    # Convert categorical features to category type
    X['waterfront'] = X['waterfront'].astype('category')
    X['view'] = X['view'].astype('category')
    X['condition'] = X['condition'].astype('category')
    X['grade'] = X['grade'].astype('category')

    return X, y


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

    X['house_age'] = X['year'] - X['yr_built'] # Calculate house age and add as new feature
    X['is_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # Add new feature to indicate if house was renovated

    # Drop unnecessary columns
    X.drop(columns=['id','date', 'yr_built', 'yr_renovated', 'lat', 'long', 'year'], inplace=True)

    X = X.fillna(X.mean()) # Handle missing values (if any) or Nan values with mean

    # Convert categorical features to category type
    # X['waterfront'] = X['waterfront'].astype('category')
    # X['view'] = X['view'].astype('category')
    # X['condition'] = X['condition'].astype('category')
    # X['grade'] = X['grade'].astype('category')

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
        # i want to save it in my mac in IML folder in Graphs-ex1 folder
        output_path = "/Users/zvimarmor/Desktop/IML/Graphs-ex1"
        plt.savefig(f"{output_path}/{column}.png")
        plt.show()
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=52) #(52 is my lucky number)

    # Question 3 - preprocessing of housing prices train dataset
    X_train_clean, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    #feature_evaluation(X_train_clean, y_train)

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
    percentages = list(range(10, 101, 1))
    losses = []

    # Iterate over each percentage of training set
    for p in percentages:
        p_losses = []
        for i in range(10):  # Repeat 10 times for each percentage
            # Sample p% of training data
            if p == 100:
                X_train_sampled, _, y_train_sampled, _ = X_train_clean, None, y_train, None
            else:
                X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train_clean, y_train, train_size=float(p)/100.0, random_state=None)

            # Fit linear regression model
            model = LinearRegression(include_intercept=True)
            model.fit(X_train_sampled, y_train_sampled)

            # Evaluate model and calculate loss
            test_loss = model.loss(X_test_clean, y_test)
            p_losses.append(test_loss)

        # Store mean and standard deviation of loss for this percentage
        losses.append((np.mean(p_losses), np.std(p_losses)))

    # Extract mean losses and standard deviations
    mean_losses = [loss[0] for loss in losses]
    std_losses = [loss[1] for loss in losses]

    # Plot mean loss as a function of percentage
    plt.errorbar(percentages, mean_losses, yerr=2 * np.array(std_losses), fmt='-o')
    plt.xlabel('Percentage of Training Set (%)')
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss vs. Percentage of Training Set')
    plt.grid(True)
    plt.show()



                      
                