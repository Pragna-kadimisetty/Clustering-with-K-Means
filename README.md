Customer Segmentation using K-Means Clustering

---------------
Project Overview

This project applies K-Means clustering to segment customers based on their annual income and spending score using the Mall Customers dataset. The goal is to group customers into distinct segments to enable better marketing strategies and personalized services.

-----------
Dataset

File: Mall_Customers.csv

CustomerID – Unique customer identifier

Gender – Male/Female

Age – Customer’s age

Annual Income (k$) – Annual income in thousands of dollars

Spending Score (1–100) – A score assigned based on customer behavior and spending

----------
Steps Performed

1.Data Loading & Preprocessing

2.Imported dataset using pandas

3.Selected relevant features for clustering

4.Optional PCA for visualization when dealing with more than two features

5.Elbow Method to determine optimal number of clusters

6.K-Means Clustering model training and prediction

7.Visualization of customer clusters with centroids

Evaluation using Silhouette Score to measure cluster quality

--------------
Libraries

Pandas – Data handling

Matplotlib – Data visualization

Scikit-learn – Machine learning (K-Means, PCA, Silhouette Score)
