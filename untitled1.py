# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import numpy as np


# Define a function to clean the World Bank data
def clean_data(filename):
    # Read the CSV file and set 'Country Name' and 'Indicator Name' as index
    df = pd.read_csv(filename, index_col=['Country Name', 'Indicator Name'])
    # Drop the unnecessary columns 'Country Code' and 'Indicator Code'
    df = df.drop(columns=['Country Code', 'Indicator Code'])
    # Convert the data to numeric type, and replace any missing values using forward and backward fill
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
    # Reset the index of the DataFrame
    df = df.reset_index()
    return df

# Call the function to clean the data and store it in a variable
world_bank_data = clean_data('mainfile.csv')

# Transpose the data
transposed_data = world_bank_data.set_index(['Country Name', 'Indicator Name']).transpose()
print(transposed_data)

# Define the indicators of interest
co2_emissions = 'CO2 emissions (kt)'
forest_area = 'Forest area (sq. km)'

# Get the statistics for the selected indicators
co2_emissions_stats = world_bank_data[world_bank_data['Indicator Name'] == co2_emissions]
forest_area_stats = world_bank_data[world_bank_data['Indicator Name'] == forest_area]


# Normalize the data
selected_data = pd.concat([co2_emissions_stats, forest_area_stats], axis=0).pivot(index='Country Name', columns='Indicator Name', values='2018')
normalized_data = (selected_data - selected_data.mean()) / selected_data.std()

# Perform clustering using k-means
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(normalized_data)

# Define a function for curve fitting
def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit the curve to the data
indicators_of_interest = ['CO2 emissions (kt)', 'Forest area (sq. km)']
x_data = normalized_data[indicators_of_interest[0]].values
y_data = normalized_data[indicators_of_interest[1]].values
popt, _ = curve_fit(exponential_func, x_data, y_data)

# Plot the cluster membership and cluster centers
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c']
for i in range(n_clusters):
    cluster_points = normalized_data.iloc[clusters == i, :]
    plt.scatter(cluster_points[indicators_of_interest[0]], cluster_points[indicators_of_interest[1]], color=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='k', marker='X', label='Cluster Centers')
plt.title('Clusters of CO2 Emissions and forest_area')
plt.xlabel('Normalized CO2 emissions per capita')
plt.ylabel('Normalized forest_area')
plt.legend()

# Plot the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(normalized_data[indicators_of_interest[0]], normalized_data[indicators_of_interest[1]], color='b', label='Data Points')
plt.plot(x_data, exponential_func(x_data, *popt), color='r', label='Curve Fit')
plt.title('Curve Fitting of CO2 Emissions and forest_area')
plt.xlabel('Normalized CO2 emissions per capita')
plt.ylabel('Normalized forest_area')
plt.legend()

plt.title('Clusters of CO2 Emissions and forest_area')