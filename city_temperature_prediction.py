
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from polynomial_fitting import PolynomialFitting
import matplotlib.pyplot as plt


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Load dataset
    X = pd.read_csv(filename, parse_dates=["Date"])

    # Drop rows with missing values
    X.dropna(inplace=True)

    # Add DayOfYear column
    X["DayOfYear"] = X["Date"].dt.dayofyear

    return X

if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    Israel_data = df[df['Country'] == 'Israel'] 
    years = Israel_data['Year'].unique()
    plt.figure(figsize=(10, 6)) # Plot scatter plot
    for year in years:
        year_data = Israel_data[Israel_data['Year'] == year]
        plt.scatter(year_data['DayOfYear'], year_data['Temp'], label=str(year))
    plt.xlabel('Day of Year')
    plt.ylabel('Temperature')
    plt.title('Temperature Variation in Israel by Day of Year')
    plt.legend(title='Year')
    #plt.show()


    # Group samples by month and calculate standard deviation of daily temperatures
    monthly_std = Israel_data.groupby('Month')['Temp'].std()
    plt.figure(figsize=(10, 6))
    monthly_std.plot(kind='bar', color='blue')
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation of Temperature (°C)')
    plt.title('Standard Deviation of Daily Temperatures by Month')
    #plt.show()

    # Question 4 - Exploring differences between countries
    countries = df['Country'].unique()
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'orange']
    for country in countries:
        country_data = df[df['Country'] == country]
        monthly_avg = country_data.groupby('Month')['Temp'].mean()
        monthly_std = country_data.groupby('Month')['Temp'].std()
        plt.errorbar(monthly_avg.index, monthly_avg, yerr=monthly_std, label=country, color=colors.pop(0))
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    plt.title('Average Monthly Temperature by Country')
    plt.legend(title='Country')
    #plt.show()

    # Question 5 - Fitting model for different values of `k`

    # Print the test error recorded for each value of k. In addition plot a bar plot showing the test error recorded for each value of k
    X, y = Israel_data.drop('Temp', axis=1), Israel_data.Temp
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=52)
    test_errors = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train[['DayOfYear']], y_train)
        test_error = model.loss(X_test[['DayOfYear']], y_test)
        test_errors.append(round(test_error, 2))
        print(f'Test error for k={k}: {test_error}')

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), test_errors)
    plt.xlabel('Degree of Polynomial (k)')
    plt.ylabel('Test Error')
    plt.title('Test Error for Different Values of k')
    plt.show()


    # Question 6 - Evaluating fitted model on different countries

    pass
