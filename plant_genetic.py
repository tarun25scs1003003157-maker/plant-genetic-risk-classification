# Plant Genetic Risk Classification Project

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans


# -------- Step 1: Create Dataset --------
data = {
    "Gene_Length": [1200, 1500, 800, 2000, 500, 1600, 900, 2200, 1800, 700],
    "Mutation_Rate": [0.2, 0.1, 0.5, 0.05, 0.6, 0.15, 0.4, 0.03, 0.07, 0.55],
    "Trait_Strength": [7, 9, 4, 10, 3, 8, 5, 10, 9, 2]
}

df = pd.DataFrame(data)

# -------- Step 2: Create Labels --------
labels = []
for i in range(len(df)):
    if df["Mutation_Rate"][i] > 0.3 or df["Trait_Strength"][i] < 5:
        labels.append("Risky")
    else:
        labels.append("Safe")

df["Label"] = labels

print("Dataset:\n")
print(df)


# -------- Step 3: Prepare Data --------
X = df[["Gene_Length", "Mutation_Rate", "Trait_Strength"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)


# -------- Step 4: Train Model --------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# -------- Step 5: Test Model --------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# -------- Step 6: Visualization --------
df["Label"].value_counts().plot(kind='bar')
plt.title("Plant Category Count")
plt.xlabel("Type")
plt.ylabel("Number")
plt.show()


# -------- Step 7: Clustering --------
kmeans = KMeans(n_clusters=2)
df["Cluster"] = kmeans.fit_predict(X)

print("\nClustering Result:\n")
print(df)


# -------- Step 8: Scatter Plot --------
plt.scatter(df["Gene_Length"], df["Mutation_Rate"], c=df["Cluster"])
plt.xlabel("Gene Length")
plt.ylabel("Mutation Rate")
plt.title("Plant Clusters")
plt.show()


# -------- Step 9: Test with New Input --------
sample = pd.DataFrame({
    "Gene_Length": [1400],
    "Mutation_Rate": [0.2],
    "Trait_Strength": [8]
})

result = model.predict(sample)

print("\nNew Plant Prediction:", result[0])