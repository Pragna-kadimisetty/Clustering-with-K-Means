import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features for clustering (you can change as needed)
X = df.iloc[:, [3, 4]].values  # Example: Annual Income & Spending Score

# Create folder for saving plots
output_dir = "kmeans_plots"
os.makedirs(output_dir, exist_ok=True)

# Optional PCA for 2D visualization (if you have more than 2 features)
if X.shape[1] > 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
else:
    X_pca = X

# 1️⃣ Elbow Method to find optimal K
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.savefig(os.path.join(output_dir, "elbow_method.png"))
plt.close()

# 2️⃣ Fit K-Means with chosen K (based on elbow method)
optimal_k = 5  # Change if elbow method suggests another
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X)

# 3️⃣ Visualize clusters
plt.figure(figsize=(8, 5))
for cluster in range(optimal_k):
    plt.scatter(X_pca[labels == cluster, 0], X_pca[labels == cluster, 1], label=f"Cluster {cluster+1}")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title("Customer Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig(os.path.join(output_dir, "clusters.png"))
plt.close()

# 4️⃣ Silhouette Score (higher = better)
score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.3f}")

print(f"✅ All plots saved in folder: {output_dir}")
