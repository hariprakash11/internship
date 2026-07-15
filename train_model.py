# ============================================================
# EduPro Recommendation System
# train_model.py
# Part 1 - Import Libraries & Load Dataset
# ============================================================

import os
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================================
# LOAD EXCEL FILE
# ============================================================

FILE_PATH = "data/EduPro Online Platform.xlsx"

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

users = pd.read_excel(FILE_PATH, sheet_name="Users")
teachers = pd.read_excel(FILE_PATH, sheet_name="Teachers")
courses = pd.read_excel(FILE_PATH, sheet_name="Courses")
transactions = pd.read_excel(FILE_PATH, sheet_name="Transactions")

print("\nDataset Loaded Successfully\n")

print("Users :", users.shape)
print("Teachers :", teachers.shape)
print("Courses :", courses.shape)
print("Transactions :", transactions.shape)

# ============================================================
# DATA CLEANING
# ============================================================

users = users.drop_duplicates()
teachers = teachers.drop_duplicates()
courses = courses.drop_duplicates()
transactions = transactions.drop_duplicates()

users = users.fillna("Unknown")
teachers = teachers.fillna("Unknown")
courses = courses.fillna("Unknown")
transactions = transactions.fillna("Unknown")

print("\nMissing values handled.")
print("Duplicate rows removed.")

# ============================================================
# MERGE DATASETS
# ============================================================

print("\nMerging datasets...")

data = transactions.merge(
    users,
    on="UserID",
    how="left"
)

data = data.merge(
    courses,
    on="CourseID",
    how="left"
)

data = data.merge(
    teachers,
    on="TeacherID",
    how="left",
    suffixes=("_Course", "_Teacher")
)

print("Merged Shape :", data.shape)

print("\nColumns Available:\n")
print(data.columns.tolist())

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating Student Features...")

student_features = pd.DataFrame()

student_features["UserID"] = users["UserID"]

# Age
student_features["Age"] = users["Age"]

# Gender
student_features["Gender"] = users["Gender"]

# Total Spending
student_features["TotalSpent"] = (
    data.groupby("UserID")["Amount"]
    .sum()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

# Average Spending
student_features["AverageSpent"] = (
    data.groupby("UserID")["Amount"]
    .mean()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

# Number of Courses Purchased
student_features["CoursesEnrolled"] = (
    data.groupby("UserID")["CourseID"]
    .count()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

# Average Course Rating
student_features["AverageRating"] = (
    data.groupby("UserID")["CourseRating"]
    .mean()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

print(student_features.head())


# ============================================================
# FEATURE ENGINEERING (Continued)
# ============================================================

print("\nCreating Advanced Features...")

# Favorite Course Category
student_features["FavoriteCategory"] = (
    data.groupby("UserID")["CourseCategory"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reindex(users["UserID"])
    .fillna("Unknown")
    .values
)

# Favorite Course Level
student_features["FavoriteLevel"] = (
    data.groupby("UserID")["CourseLevel"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reindex(users["UserID"])
    .fillna("Unknown")
    .values
)

# Preferred Course Type
student_features["PreferredType"] = (
    data.groupby("UserID")["CourseType"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reindex(users["UserID"])
    .fillna("Unknown")
    .values
)

# Preferred Payment Method
student_features["PaymentMethod"] = (
    data.groupby("UserID")["PaymentMethod"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reindex(users["UserID"])
    .fillna("Unknown")
    .values
)

# Average Teacher Rating
student_features["TeacherRating"] = (
    data.groupby("UserID")["TeacherRating"]
    .mean()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

# Average Teacher Experience
student_features["TeacherExperience"] = (
    data.groupby("UserID")["YearsOfExperience"]
    .mean()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

print("\nFeature Engineering Completed")
print(student_features.head())

# ============================================================
# LABEL ENCODING
# ============================================================

print("\nEncoding Categorical Features...")

categorical_columns = [
    "Gender",
    "FavoriteCategory",
    "FavoriteLevel",
    "PreferredType",
    "PaymentMethod"
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    student_features[column] = encoder.fit_transform(
        student_features[column].astype(str)
    )
    encoders[column] = encoder

print("Categorical Encoding Completed")

# ============================================================
# FEATURE MATRIX
# ============================================================

feature_columns = [
    "Age",
    "Gender",
    "TotalSpent",
    "AverageSpent",
    "CoursesEnrolled",
    "AverageRating",
    "FavoriteCategory",
    "FavoriteLevel",
    "PreferredType",
    "PaymentMethod",
    "TeacherRating",
    "TeacherExperience"
]

X = student_features[feature_columns]

print("\nFeature Matrix Shape:", X.shape)

# ============================================================
# STANDARDIZATION
# ============================================================

print("\nScaling Features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaling Completed")


# ============================================================
# ELBOW METHOD
# ============================================================

print("\nFinding Optimal Number of Clusters...")

wcss = []

cluster_range = range(2, 11)

for k in cluster_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    cluster_range,
    wcss,
    marker="o",
    linewidth=2
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.savefig(
    "outputs/elbow_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Elbow Curve Saved")

# ============================================================
# TRAIN KMEANS
# ============================================================

print("\nTraining KMeans Model...")

NUM_CLUSTERS = 4

kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init=10
)

student_features["Cluster"] = kmeans.fit_predict(
    X_scaled
)

print("KMeans Training Completed")

# ============================================================
# SILHOUETTE SCORE
# ============================================================

score = silhouette_score(
    X_scaled,
    student_features["Cluster"]
)

print(f"\nSilhouette Score : {score:.4f}")

with open(
    "outputs/silhouette_score.txt",
    "w"
) as file:

    file.write(
        f"Silhouette Score : {score:.4f}"
    )

# ============================================================
# PCA VISUALIZATION
# ============================================================

print("\nPerforming PCA...")

pca = PCA(n_components=2)

components = pca.fit_transform(X_scaled)

student_features["PC1"] = components[:,0]

student_features["PC2"] = components[:,1]

plt.figure(figsize=(8,6))

plt.scatter(

    student_features["PC1"],

    student_features["PC2"],

    c=student_features["Cluster"],

    cmap="viridis",

    s=70

)

plt.title("Student Clusters (PCA)")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.grid(True)

plt.savefig(

    "outputs/pca_clusters.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("PCA Plot Saved")

# ============================================================
# SAVE TRAINED MODELS
# ============================================================

print("\nSaving Models...")

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    kmeans,
    "models/kmeans.pkl"
)

joblib.dump(
    pca,
    "models/pca.pkl"
)

print("Models Saved Successfully")

# ============================================================
# SAVE STUDENT SEGMENTS
# ============================================================

student_features.to_csv(
    "outputs/student_segments.csv",
    index=False
)

print("Student Segments Saved")

# ============================================================
# CLUSTER SUMMARY
# ============================================================

summary = student_features.groupby("Cluster").agg({

    "Age":"mean",

    "TotalSpent":"mean",

    "AverageSpent":"mean",

    "CoursesEnrolled":"mean",

    "AverageRating":"mean",

    "TeacherRating":"mean",

    "TeacherExperience":"mean"

}).round(2)

summary.to_csv(

    "outputs/cluster_summary.csv"

)

print("\nCluster Summary")

print(summary)

# ============================================================
# CLUSTER COUNTS
# ============================================================

print("\nStudents Per Cluster")

print(

    student_features["Cluster"]

    .value_counts()

    .sort_index()

)

# ============================================================
# FEATURE IMPORTANCE (Summary Only)
# ============================================================

feature_summary = pd.DataFrame({

    "Feature": feature_columns,

    "Mean": X.mean().values,

    "Std": X.std().values

})

feature_summary.to_csv(

    "outputs/feature_summary.csv",

    index=False

)

# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")

print("✔ models/scaler.pkl")
print("✔ models/kmeans.pkl")
print("✔ models/pca.pkl")

print("\nOutputs:")

print("✔ outputs/student_segments.csv")
print("✔ outputs/cluster_summary.csv")
print("✔ outputs/feature_summary.csv")
print("✔ outputs/elbow_curve.png")
print("✔ outputs/pca_clusters.png")
print("✔ outputs/silhouette_score.txt")

print("\nYour dataset has been clustered successfully!")
print("=" * 60)

joblib.dump(encoders, "models/encoders.pkl")
print("✔ models/encoders.pkl")