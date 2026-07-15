import pandas as pd
import joblib

print("=" * 60)
print("Loading Models...")
print("=" * 60)

scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")
encoders = joblib.load("models/encoders.pkl")

print("Models Loaded Successfully")

FILE_PATH = "data/EduPro Online Platform.xlsx"

print("\nLoading Dataset...")

users = pd.read_excel(FILE_PATH, sheet_name="Users")
courses = pd.read_excel(FILE_PATH, sheet_name="Courses")
teachers = pd.read_excel(FILE_PATH, sheet_name="Teachers")
transactions = pd.read_excel(FILE_PATH, sheet_name="Transactions")

print("Dataset Loaded")

# ----------------------------
# Merge
# ----------------------------

data = transactions.merge(users, on="UserID", how="left")
data = data.merge(courses, on="CourseID", how="left")
data = data.merge(
    teachers,
    on="TeacherID",
    how="left",
    suffixes=("_Course", "_Teacher")
)

print("Merge Completed")

# ----------------------------
# Feature Engineering
# ----------------------------

student_features = pd.DataFrame()

student_features["UserID"] = users["UserID"]
student_features["Age"] = users["Age"]
student_features["Gender"] = users["Gender"]

student_features["TotalSpent"] = (
    data.groupby("UserID")["Amount"]
    .sum()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

student_features["AverageSpent"] = (
    data.groupby("UserID")["Amount"]
    .mean()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

student_features["CoursesEnrolled"] = (
    data.groupby("UserID")["CourseID"]
    .count()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

student_features["AverageRating"] = (
    data.groupby("UserID")["CourseRating"]
    .mean()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

student_features["FavoriteCategory"] = (
    data.groupby("UserID")["CourseCategory"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reindex(users["UserID"])
    .fillna("Unknown")
    .values
)

student_features["FavoriteLevel"] = (
    data.groupby("UserID")["CourseLevel"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reindex(users["UserID"])
    .fillna("Unknown")
    .values
)

student_features["PreferredType"] = (
    data.groupby("UserID")["CourseType"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reindex(users["UserID"])
    .fillna("Unknown")
    .values
)

student_features["PaymentMethod"] = (
    data.groupby("UserID")["PaymentMethod"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
    .reindex(users["UserID"])
    .fillna("Unknown")
    .values
)

student_features["TeacherRating"] = (
    data.groupby("UserID")["TeacherRating"]
    .mean()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

student_features["TeacherExperience"] = (
    data.groupby("UserID")["YearsOfExperience"]
    .mean()
    .reindex(users["UserID"])
    .fillna(0)
    .values
)

# ----------------------------
# Encode
# ----------------------------

categorical_columns = [
    "Gender",
    "FavoriteCategory",
    "FavoriteLevel",
    "PreferredType",
    "PaymentMethod"
]

for col in categorical_columns:
    student_features[col] = encoders[col].transform(
        student_features[col].astype(str)
    )

# ----------------------------
# Scale
# ----------------------------

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

X_scaled = scaler.transform(X)

student_features["Cluster"] = kmeans.predict(X_scaled)

print("\nStudent Segments")
print(student_features[["UserID", "Cluster"]].head())

# ----------------------------
# Recommendations
# ----------------------------

print("\nSample Recommendations\n")

for user in student_features["UserID"].head(10):

    cluster = student_features.loc[
        student_features["UserID"] == user,
        "Cluster"
    ].iloc[0]

    category = student_features.loc[
        student_features["UserID"] == user,
        "FavoriteCategory"
    ].iloc[0]

    recommended = courses[
        courses["CourseCategory"] == encoders["FavoriteCategory"].inverse_transform([category])[0]
    ].head(5)

    print(f"\nUser : {user}")
    print(f"Cluster : {cluster}")

    print(recommended[[
        "CourseID",
        "CourseName",
        "CourseCategory",
        "CourseLevel"
    ]])