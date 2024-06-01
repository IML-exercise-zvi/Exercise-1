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
    
    X['DayOfYear'] = X['Date'].dt.dayofyear # Add a 'DayOfYear' column based on the 'Date' column

    return X

if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    Israel_data = df[df['Country'] == 'Israel'] 

    #remove all data points where temperature is -72.77777777777777
    Israel_data = Israel_data[Israel_data['Temp'] != -72.77777777777777]

    years = Israel_data['Year'].unique()
    plt.figure(figsize=(10, 6)) # Plot scatter plot
    for year in years:
        year_data = Israel_data[Israel_data['Year'] == year]
        plt.scatter(year_data['DayOfYear'], year_data['Temp'], label=str(year))
    plt.xlabel('Day of Year')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Variation in Israel by Day of Year')
    plt.legend(title='Year')
    plt.savefig('temperature_variation_by_day_of_year.png')

    # Group samples by month and calculate standard deviation of daily temperatures
    monthly_std = Israel_data.groupby('Month')['Temp'].std()
    plt.figure(figsize=(10, 6))
    monthly_std.plot(kind='bar', color='blue')
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation of Temperature (°C)')
    plt.title('Standard Deviation of Daily Temperatures by Month')
    plt.savefig('temperature_std_by_month.png')

    # Question 4 - Exploring differences between countries
    countries = df['Country'].unique()
    plt.figure(figsize=(10, 6))

    for country in countries:
        if country == 'Israel':
            country_data = Israel_data
        else:
            country_data = df[df['Country'] == country]
        monthly_mean = country_data.groupby('Month')['Temp'].mean()
        monthly_std = country_data.groupby('Month')['Temp'].std()
        plt.plot(monthly_mean.index, monthly_mean, label=country)
        plt.errorbar(monthly_mean.index, monthly_mean, yerr=monthly_std, fmt='o')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (°C)')
    plt.title('Average Monthly Temperature by Country')
    plt.legend(title='Country')
    plt.savefig('average_monthly_temperature_by_country.png')

    # Question 5 - Fitting model for different values of `k`
    X, y = Israel_data['DayOfYear'].values, Israel_data['Temp'].values #create design "matrix" (actually a vector) and response vector

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 52) #split data into training and testing sets

    test_errors = []

    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train, y_train) #fit model for each k
        test_error = model.loss(X_test, y_test) #calculate test error for each k
        test_errors.append(round(test_error, 2))
        print(f'Test error for k={k}: {round(test_error, 2)}')

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), test_errors)
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Mean Squared Error')
    plt.title('Test Error of Polynomial Fitting')
    plt.savefig('test_error_of_polynomial_fitting_by_k.png')

    # Question 6 - Evaluating fitted model on different countries
    model = PolynomialFitting(5) #fit model with k=5 (because it had the lowest test error)
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
    plt.savefig('test_error_of_polynomial_fitting_by_country.png')

