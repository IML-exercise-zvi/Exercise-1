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

    X.dropna(inplace=True) # Drop rows with missing values
    X['Date'] = pd.to_datetime(X['Date']) # Convert 'Date' column to datetime
    
    # Add a 'DayOfYear' column based on the 'Date' column
    X['DayOfYear'] = X['Date'].dt.dayofyear

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
    plt.show()

    # Group samples by month and calculate standard deviation of daily temperatures
    monthly_std = Israel_data.groupby('Month')['Temp'].std()
    plt.figure(figsize=(10, 6))
    monthly_std.plot(kind='bar', color='blue')
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation of Temperature (°C)')
    plt.title('Standard Deviation of Daily Temperatures by Month')
    plt.show()

    # Question 4 - Exploring differences between countries
    countries = df['Country'].unique() # Get unique countries
    plt.figure(figsize=(10, 6))

    for country in countries:
        country_data = df[df['Country'] == country]
        monthly_mean = country_data.groupby('Month')['Temp'].mean()
        monthly_std = country_data.groupby('Month')['Temp'].std()
        plt.plot(monthly_mean.index, monthly_mean, label=country)
        plt.fill_between(monthly_mean.index, monthly_mean - monthly_std, monthly_mean + monthly_std, alpha=0.1) # Fill between mean-std and mean+std
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (°C)')
    plt.title('Average Monthly Temperature by Country')
    plt.legend(title='Country')
    plt.show()

    # Question 5 - Fitting model for different values of `k`
    X, y = Israel_data['DayOfYear'].values, Israel_data['Temp'].values #create design "matrix" (actually a vector) and response vector

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split data into training and testing sets

    # Remove outliers
    mask = y != -72.77777777777777
    X, y = X[mask], y[mask]

    test_errors = []

    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        test_error = model.loss(X_test, y_test) #calculate test error for each k
        test_errors.append(round(test_error, 2))
        print(f'Test error for k={k}: {test_error}')

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), test_errors)
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Mean Squared Error')
    plt.title('Test Error of Polynomial Fitting')
    plt.show()

    # Question 6 - Evaluating fitted model on different countries
    model = PolynomialFitting(6)
    model.fit(X_train, y_train)
    
    countries = [country for country in countries if country != 'Israel']
    errors = []
    for country in countries: #calculate test error for each country besides israel
        country_data = df[df['Country'] == country]
        X, y = country_data['DayOfYear'].values, country_data['Temp'].values
        error = model.loss(X, y) 
        errors.append(round(error, 2))

    plt.figure(figsize=(10, 6))
    plt.bar(countries, errors)
    plt.xlabel('Country')
    plt.ylabel('Mean Squared Error')
    plt.title('Test Error of Polynomial Fitting on Different Countries')
    plt.show()

