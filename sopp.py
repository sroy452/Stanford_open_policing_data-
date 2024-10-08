# -*- coding: utf-8 -*-
"""SOPP.ipynb

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""Read CSV file"""

df = pd.read_csv("traindata.csv")
df.head(10)

from google.colab import drive
drive.mount('/content/drive')

df.info()

"""Data checks"""

df.isna().sum()

"""See number of rows and columns"""

df.shape

"""Description"""

df.describe()

"""Lots of empty county names, no use"""

df = df.drop(columns=["county_name"])

"""Since driver gender is key to analysis, and we can see rows without gender don't have other stuff as well, let's drop rows which don't have it listed"""

df = df.dropna(axis=0, subset=["driver_gender"])

"""Let's get a look at our clean(ed) data


"""

df.isnull().sum()

df.shape

"""Let's see the curious cases of violation and violation_raw"""

df["violation"].unique()

df["violation_raw"].unique()

"""Stop Timings"""

df.stop_time.dtype

df['stop_time']=pd.to_datetime(df.stop_time,format='%H:%M')
df['stop_hour']=df['stop_time'].dt.hour

df['stop_hour'].value_counts()

"""Visualize"""

plt.figure(figsize=(10,8))
sns.countplot(y='stop_hour',data=df,)

"""Stop Duration"""

df['stop_duration'].value_counts()

mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
df['stop_minutes'] = df.stop_duration.map(mapping)
df.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])

"""Time vs type breakdown"""

sns.catplot(x="stop_duration", data=df, hue="violation_raw", kind="count")

"""Visualize stop time for different violations"""

df.groupby(by='violation_raw').stop_minutes.mean().plot(kind='barh',color='red')

"""Let's see what races we're working with"""

print(df["driver_race"].unique())
len(df["driver_race"].unique())

"""Let's check out Date range"""

df['stop_date'] = pd.to_datetime(df['stop_date'])

min_date = df['stop_date'].min()
print("Minimum date: ", min_date)
max_date = df['stop_date'].max()
print("Maximum date: ", max_date)
date_range = max_date - min_date
print("Date range: ", date_range)

"""Time for age- must see"""

bin_edges = [0, 17, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 76, 110]
bin_labels = ['Less than 18', '18 - 25', '26 - 30', '31 - 35', '36 - 40','41 - 45', '46 - 50', '51 - 55', '56 - 60', '61 - 65', '66 - 70', '71 - 75', 'over 76']
df['age_break_down'] = pd.cut(df['driver_age'], bin_edges, labels=bin_labels)
df.groupby(df['age_break_down']).size()

"""Relation between age and violation"""

df.groupby(by='violation').driver_age.describe()

"""Visualize graphically"""

plt.figure(figsize = (15,6))
sns.histplot(df.driver_age, kde = True)
plt.xticks(rotation=0);

"""Male and female subdivisions"""

female = df[df.driver_gender == 'F']
male = df[df.driver_gender == 'M']

plt.figure(figsize = (15,6))
sns.histplot(male.driver_age, color = '#1AA7EC', kde = True);

plt.figure(figsize = (15,6))
sns.histplot(female.driver_age, color = '#FFC0CB', kde = True);

"""Again, to avoid scrolling, relist the table"""

df.head()

"""Let's see the outcomes of stops"""

counts = df['stop_outcome'].value_counts()
counts

"""Time to get icky into race"""

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

counts = df['driver_race'].value_counts()
dictionary = dict(counts)

print(dictionary)
items = []
values = []

for _ in range(len(counts)):
    for item in dictionary:
        value = dictionary[item]
        items.append(item)
        values.append(value)

fig, ax = plt.subplots()
ax.barh(items, values)

labels = ax.get_xticklabels()


ax.set_title('Races caught in traffic stops')
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')


plt.show()

"""Traffic stops caught on record"""

import matplotlib.pyplot as plt
most_common_outcome = df.groupby(['driver_gender', 'stop_outcome']).size().unstack().fillna(0).plot(kind="bar")

most_common_outcome = df.groupby(['driver_gender', 'stop_outcome', 'driver_race' ]).size().unstack().fillna(0)
most_common_outcome.plot(kind='bar', stacked=True)

"""Checking Traffic stop counts by Race"""

ethnic_groups = df['driver_race'].value_counts()
print(ethnic_groups.sum())
ethnic_groups


# Create a stacked bar plot
ethnic_groups.plot(kind='bar',  stacked=False)
plt.xlabel('Race and ethnicity')
plt.ylabel('Counts')
plt.title('Traffic stop counts by Race and ethnicity')
plt.xticks(rotation=45, ha='right')
formatter = ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()

"""Let's set up the heatmap (correlation matrix)"""

import seaborn as sns

sns.set(style="white")
# Compute the correlation matrix
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(27,18))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(15,963 ,as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation of Attirubtues",fontsize=30,fontweight='bold')
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)

"""Do men or women speed more often?"""

df.driver_gender.value_counts()

plt.figure(figsize = (15,6))
plt.pie(df.driver_gender.value_counts(),
        labels =df.driver_gender.value_counts().index,
        colors = ['#1AA7EC', '#FFC0CB'],
       autopct='%.1f%%');

"""When a man is pulled over, what's generally the reason?"""

df[df.driver_gender == "M"].violation.value_counts(normalize=True)

"""When a woman is pulled over, what's generally the reason?"""

df[df.driver_gender == "F"].violation.value_counts(normalize=True)

"""Graphics"""

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df[df.driver_gender == "F"].violation.value_counts(normalize=True).plot(kind="bar")
plt.title("Violation of Women")

plt.subplot(2, 2, 2)
df[df.driver_gender == "M"].violation.value_counts(normalize=True).plot(kind="bar")
plt.title("Violation of Men")

"""Time for some clustering"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = df[['driver_age', 'drugs_related_stop']]

# Handle missing values if needed
features = features.dropna()

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Choose the number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve to determine the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()

# Based on the elbow method, choose the optimal K (let's say 3 for this example)
optimal_k = 3

# Fit K-means model
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(features_scaled)

# Add cluster labels to the DataFrame
df['cluster'] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='driver_age', y='drugs_related_stop', hue='cluster', data=df, palette='viridis', style='cluster', s=100)
plt.title('K-means Clustering of Driver Age vs. Drugs Related Stop')
plt.xlabel('Driver Age')
plt.ylabel('Drugs Related Stop (Binary)')
plt.legend(title='Cluster', loc='upper right')
plt.show()

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Load your dataset (replace with your file path)
df = pd.read_csv("traindata.csv")

# Select relevant features
selected_features = ["driver_gender", "driver_race"]

# Handle categorical features appropriately
if df["driver_gender"].dtype == "object":
    df["driver_gender"] = pd.Categorical(df["driver_gender"]).codes  # One-hot encoding
if df["driver_race"].dtype == "object":
    df["driver_race"] = pd.Categorical(df["driver_race"]).codes  # One-hot encoding

# Choose linkage method (e.g., 'ward' for minimizing within-cluster variance)
linkage = 'ward'

# Instantiate and fit Hierarchical Clustering model
hc = AgglomerativeClustering(n_clusters=3, linkage=linkage)  # Replace ... with your chosen number of clusters (3 in this case)
hc_labels = hc.fit_predict(df[selected_features])

# Add cluster labels to the DataFrame
df["cluster_labels"] = hc_labels

# View cluster distributions
print(df.groupby("cluster_labels").describe())

import matplotlib.pyplot as plt

# Assign cluster colors
cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}  # Adjust colors as needed
colors = [cluster_colors[label] for label in df['cluster_labels']]

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['driver_gender'], df['driver_race'], c=colors, s=50)
plt.xlabel("driver_gender")
plt.ylabel("driver_race")
plt.title("Hierarchical Clustering Scatter Plot")
plt.show()
