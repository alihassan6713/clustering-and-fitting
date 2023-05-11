import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Set the background color to black
plt.style.use('dark_background')

# Load the climate change data from the World Bank dataset
data_url = 'mainfile.csv'
df = pd.read_csv(data_url)

# Remove missing values
data = df.dropna()

# Select relevant columns for analysis
selected_columns = ['Country Name', 'Indicator Name', '2019']
data = df[selected_columns]

# Pivot the data to have indicators as columns
pivoted_data = data.pivot(index='Country Name', columns='Indicator Name', values='2019')

# Select specific indicators for analysis
indicators_of_interest = ['CO2 emissions (metric tons per capita)', 'Population, total']
selected_data = pivoted_data[indicators_of_interest]

# Remove missing values
selected_data = selected_data.dropna()

# Normalize the data
normalized_data = (selected_data - selected_data.mean()) / selected_data.std()

# Perform clustering using k-means
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(normalized_data)

# Define a function for curve fitting
def exponential_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit the curve to the data
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
plt.title('Clusters of CO2 Emissions and Population Growth')
plt.xlabel('Normalized CO2 emissions per capita')
plt.ylabel('Normalized Population Growth')
plt.legend()

# Save the cluster plot as an image
plt.savefig('cluster_plot.png',dpi=300)

# Plot the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(normalized_data[indicators_of_interest[0]], normalized_data[indicators_of_interest[1]], color='b', label='Data Points')
plt.plot(x_data, exponential_func(x_data, *popt), color='r', label='Curve Fit')
plt.title('Curve Fitting of CO2 Emissions and Population Growth')
plt.xlabel('Normalized CO2 emissions per capita')
plt.ylabel('Normalized Population Growth')
plt.legend()

# Save the curve fit plot as an image
plt.savefig('curve_fit_plot.png',dpi=300)

# Create correlation matrix
corr_matrix = selected_data.corr()

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')

# Save the correlation heatmap as an image
plt.savefig('correlation_heatmap.png',dpi=300)


