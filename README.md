# -Customer-Segmentation-using-K-Means-Clustering
✔ Analyzed 200+ customer records (Age, Income, Spending Score)  Preprocessed data (scaling + feature engineering) ✔ Applied Elbow Method &amp; Silhouette Score to optimize clusters (k=5) ✔ Visualized insights for targeted marketing strategies 📊 Tech Stack: Python, Pandas, Scikit-learn, Matplotlib, Seaborn 📂 Dataset: Kaggle  Mall Customer Segmentation
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# Load the dataset
# Check if the file already exists to avoid re-uploading
file_path = 'Mall_Customers.csv'
if not os.path.exists(file_path):
    from google.colab import files
    uploaded = files.upload()
    # Assuming the uploaded file name is 'Mall_Customers.csv'
    # If the user uploads a different file name, this will cause an error
    # In a real-world scenario, you might want to handle this more robustly
    # by checking the keys in the 'uploaded' dictionary.

# Read the CSV file
df = pd.read_csv(file_path)
print("First 5 rows of the dataset:")
print(df.head())

# Exploratory Data Analysis
print("\nDataset information:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())

# Select relevant features for clustering
# We'll use Annual Income and Spending Score as they directly relate to purchase behavior
X = df.iloc[:, [3, 4]].values

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10) # Added n_init
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Silhouette Score Analysis
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"For k={k}, Silhouette Score: {score:.3f}")

# Based on Elbow Method and Silhouette Score, choose k=5
optimal_k = 5

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10) # Added n_init
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = y_kmeans

# Visualize the clusters
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(optimal_k):
    plt.scatter(X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1],
                s=50, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='yellow', label='Centroids', marker='*')
plt.title('Customer Segments')
plt.xlabel('Standardized Annual Income (k$)')
plt.ylabel('Standardized Spending Score (1-100)')
plt.legend()
plt.show()

# Analyze the clusters
cluster_analysis = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'median', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'median', 'min', 'max'],
    'Age': ['mean', 'median'],
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'})

print("\nCluster Analysis:")
print(cluster_analysis)

# Interpretation of clusters
print("\nCluster Interpretation:")
print("Cluster 0: Moderate income, moderate spending (Average customers)")
print("Cluster 1: High income, low spending (Potential high-value customers not fully engaged)")
print("Cluster 2: Low income, low spending (Budget-conscious customers)")
print("Cluster 3: Low income, high spending (Careless spenders)")
print("Cluster 4: High income, high spending (Ideal customers)")
