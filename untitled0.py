import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Load the climate change data from the World Bank dataset
data_url = 'mainfile.csv'
df = pd.read_csv(data_url)

# Transpose the data
world_bank_data_t = df.transpose()
print(world_bank_data_t)

# Select relevant columns for analysis
selected_columns = ['Country Name', 'Country Code', 'Indicator Name', '2019']
data = df[selected_columns]

# Pivot the data to have indicators as columns
pivoted_data = data.pivot(index='Country Code', columns='Indicator Name', values='2019')

# Select specific indicators for analysis
indicators_of_interest = ['CO2 emissions (metric tons per capita)', 'CO2 emissions (kt)']
selected_data = pivoted_data[indicators_of_interest]

# Remove missing values
selected_data = selected_data.dropna()

# Normalize the data
normalized_data = (selected_data - selected_data.mean()) / selected_data.std()

# Perform clustering using k-means
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(normalized_data)

# Perform curve fitting using curve_fit function
# Define the fitting function (e.g., exponential growth, logistic function, polynomial)
def fitting_function(x, param1, param2):
    # Define your fitting function here
    return ...

# Fit the data to the model
params, _ = curve_fit(fitting_function, numeric_data['x'], numeric_data['y'])

# Generate predictions using the fitted model
# Define the range of x values for prediction
x_range = np.linspace(min(numeric_data['x']), max(numeric_data['x']), num=100)
predictions = fitting_function(x_range, *params)

# Fit the curve to the data
x_data = normalized_data[indicators_of_interest[0]].values
y_data = normalized_data[indicators_of_interest[1]].values
popt, _ = curve_fit(exponential_func, x_data, y_data)

# Plot the cluster membership and cluster centers
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c']
for i in range(n_clusters):
    cluster_points = normalized_data.iloc[clusters == i, :]
    plt.scatter(
        cluster_points[indicators_of_interest[0]],
        cluster_points[indicators_of_interest[1]],
        color=colors[i],
        label=f'Cluster {i+1}'
    )
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color='k',
    marker='X',
    label='Cluster Centers'
)
plt.title('Clusters of CO2 Emissions and GDP per Capita')
plt.xlabel('Normalized CO2 emissions per capita')
plt.ylabel('Normalized GDP per capita')
plt.legend()

# Plot the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(
    normalized_data[indicators_of_interest[0]],
    normalized_data[indicators_of_interest[1]],
    color='b',
    label='Data Points'
)
plt.plot(
    x_data,
    exponential_func(x_data, *popt),
    color='r',
    label='Curve Fit'
)
plt.title('Curve Fitting of CO2 Emissions and GDP per Capita')
plt.xlabel('Normalized CO2 emissions per capita')
plt.ylabel('Normalized GDP per capita')
plt.legend()

# Show the plots
plt.show()


