import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle

# Define the model function
def linear_model(x, a, b):
    return a * x + b

# Function to read the data
def read_worldbank_data(file_path):
    df = pd.read_csv(file_path, index_col='Country Name')
    df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
    dfyears = df.T
    dfyears.index.name = 'Year'
    dfyears.columns.name = 'Country Name'
    dfyears.index = pd.to_datetime(dfyears.index, format='%Y').year
    return dfyears

# Reading agriculture land data
file_path = "population.csv"
dfyears = read_worldbank_data(file_path)

# Select data from 1990 to 2022
dfyears = dfyears.loc[(dfyears.index >= 1970) & (dfyears.index <= 2022)]

# Select United Kingdom
country_data = dfyears['Italy'].dropna()
x_data = country_data.index.values
y_data = country_data.values

# Fit the model to the data
popt, pcov = curve_fit(linear_model, x_data, y_data, p0=(1, 0))

# Calculate confidence intervals
perr = np.sqrt(np.diag(pcov))
confidence_interval = [popt - perr, popt + perr]

# Predict future values
years_to_predict = np.arange(x_data.min(), 2023)
predicted_values = linear_model(years_to_predict, *popt)

# Plot the results
plt.figure(figsize=(8, 5), dpi = 300)
plt.plot(x_data, y_data, 'bo', label="Actual data")
plt.plot(years_to_predict, predicted_values, 'r-', label="Predicted data")
plt.fill_between(years_to_predict, linear_model(years_to_predict, *confidence_interval[0]), linear_model(years_to_predict, *confidence_interval[1]), color='gray', alpha=0.2)

# Add a box around the predicted data
box_start_year = 2022
rectangle = Rectangle((box_start_year, plt.ylim()[0]), 2023 - box_start_year, plt.ylim()[1]-plt.ylim()[0], fill=False, color='purple', linestyle='--')
plt.gca().add_patch(rectangle)

plt.xlabel('Year')
plt.ylabel('population')
plt.title('Italy population ')
plt.legend()
plt.show()
