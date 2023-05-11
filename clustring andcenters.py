import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics as skmet

def read_worldbank_data(filepath):
    df = pd.read_csv(filepath, index_col='Country Name')
    df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
    df_years = df.T
    df_years.index.name = 'Year'
    df_years.columns.name = 'Country Name'
    df_years.index = pd.to_datetime(df_years.index, format='%Y').year
    df_years_cols = df_years.copy()
    df_years_cols = df_years_cols.loc[:, ~df_years_cols.isna().all()]

    return df_years_cols

# Reading electricity power consumption data
file_pathcons = "Electricity power consumption.csv"
dfcoms = read_worldbank_data(file_pathcons)

# Reading electricity power transmission and distribution losses data
filepath_loss = "Electric power transmission and distribution losses.csv"
df_loss = read_worldbank_data(filepath_loss)

# Merge the two datasets on 'Year' and 'Country Name'
dfcols = pd.merge(df_loss, dfcoms, how='inner', left_index=True, right_index=True, suffixes=('_loss', '_cons'))

# Selected countries
countries = dfcols.columns.str.replace('_cons', '').str.replace('_loss', '').unique()[:81]

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Data for power losses and consumption for the year 2014
data_2014_loss = imputer.fit_transform(dfcols.loc[2014, [country+'_loss' for country in countries]].values.reshape(-1, 1))
data_2014_cons = imputer.fit_transform(dfcols.loc[2014, [country+'_cons' for country in countries]].values.reshape(-1, 1))

# Combine loss and consumption data into a single dataset
data_2014_combined = np.hstack((data_2014_loss, data_2014_cons))

# Normalize the dataframes
scaler = MinMaxScaler()
data_2014_combined = scaler.fit_transform(data_2014_combined)

# Perform KMeans clustering
nc = 4 # number of cluster centres
kmeans_2014 = KMeans(n_clusters=nc, random_state=0).fit(data_2014_combined)

# Extract labels and cluster centers
labels_2014 = kmeans_2014.labels_
cen_2014 = kmeans_2014.cluster_centers_

# Create a dataframe for plotting
df_2014_fit = pd.DataFrame(data_2014_combined, columns=['Power Loss', 'Power Consumption'], index=countries)
df_2014_fit['Cluster'] = labels_2014

# Plot the clusters
plt.figure(figsize=(10, 6),dpi = 300)
for i, country in enumerate(countries):
    cluster = df_2014_fit.loc[country, 'Cluster']
    plt.scatter(df_2014_fit.loc[country, 'Power Loss'], df_2014_fit.loc[country, 'Power Consumption'], color=plt.cm.tab10(cluster), label=country)

# plot the centers
plt.scatter(cen_2014[:, 0], cen_2014[:, 1], c="k", marker="d", s=80)
plt.title(f"{nc} Clusters in 2014")
plt.xlabel("Normalized Power Loss")
plt.ylabel("Normalized Power Consumption")
plt.show()
