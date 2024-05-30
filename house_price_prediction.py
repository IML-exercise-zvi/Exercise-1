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

    X['house_age'] = X['year'] - X['yr_built']  # Calculate house age and add as new feature
    X['is_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)  # Add new feature to indicate if house was renovated

    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())  # Handle missing values (if any) or Nan values with mean
    y = y.loc[X.index]  # connect y to X

    y_nan = np.isnan(y)
    y[y_nan] = y.fillna(y.mean())  # Handle missing values in y

    # Drop unnecessary or 'wrong' rows
    good_rows = (X['yr_built'] <= X['yr_renovated']) | (X['yr_renovated'] == 0)
    X = X[good_rows]  # Drop rows where year built is greater than year renovated
    y = y.loc[X.index]

    X = X[X['sqft_living'] > 99]  # Drop rows where sqft_living is less than 100

    # Drop unnecessary columns entirely
    X = X.drop(columns=['id', 'date', 'yr_built', 'yr_renovated', 'lat', 'long', 'year'])

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

    X['house_age'] = X['year'] - X['yr_built']  # Calculate house age and add as new feature
    X['is_renovated'] = X['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)  # Add new feature to indicate if house was renovated

    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())  # Handle missing values (if any) or Nan values with mean

    # Drop unnecessary columns entirely
    X = X.drop(columns=['id', 'date', 'yr_built', 'yr_renovated', 'lat', 'long', 'year'])

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
        #plt.savefig(f"{output_path}/{column}.png")
        plt.show()
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=52) #(52 is my lucky number)

    # Question 3 - preprocessing of housing prices train dataset
    X_train_clean, y_train_clean = preprocess_train(X_train, y_train)

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
                X_train_sampled, y_train_sampled = X_train_clean, y_train_clean
            else:
                X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train_clean, y_train_clean, train_size=float(p)/100.0, random_state=i)

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
    plt.fill_between(percentages, np.array(mean_losses) - 2 * np.array(std_losses), np.array(mean_losses) + 2 * np.array(std_losses), alpha=0.5)
    plt.plot(percentages, mean_losses)
    plt.xlabel('Percentage of Training Set (%)')
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss vs. Percentage of Training Set')
    plt.grid(True)
    plt.show()



                      
                