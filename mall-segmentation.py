# Mall Customer Segmentation using K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Create sample mall customer dataset
data = {
    "CustomerID": list(range(1, 16)),
    "Gender": [
        "Male", "Male", "Female", "Female", "Male",
        "Female", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female"
    ],
    "Age": [19, 21, 20, 23, 31, 22, 35, 40, 30, 25, 45, 32, 27, 50, 29],
    "Annual_Income": [
        15, 16, 17, 18, 25, 30, 35, 40, 45, 50,
        55, 60, 65, 70, 75
    ],
    "Spending_Score": [
        39, 81, 6, 77, 40, 76, 20, 13, 75, 35,
        45, 50, 60, 10, 85
    ]
}

df = pd.DataFrame(data)

# Select features for clustering
X = df[["Annual_Income", "Spending_Score"]]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display customer segmentation results
print("MALL CUSTOMER SEGMENTATION")
print("=" * 40)
print(df.to_string(index=False))

# Display cluster counts
print("\nNumber of customers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Visualize the customer segments
plt.figure(figsize=(9, 6))

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    s=100
)

plt.xlabel("Annual Income (in thousands)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segmentation using K-Means")
plt.grid(True, alpha=0.3)

plt.savefig("mall_customer_segmentation.png", dpi=300)
plt.show()

# Save results to CSV
import os
print("Saving CSV in:",os.getcwd())
print("TEST:code is running")
df.to_csv("mall_customer_segments.csv", index=False)
print("CSV file saved successfully!")
print(df.head())
print("\nGraph saved as mall_customer_segmentation.png")
print("\nResults saved as mall_customer_segments.csv")
print("\nCustomer segmentation completed successfully!")
