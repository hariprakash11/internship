# ============================================================
# EduPro Recommendation System
# Streamlit Dashboard
# Part 1
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="EduPro AI Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

h1,h2,h3{
    color:#003366;
}

div[data-testid="metric-container"]{
    background:#ffffff;
    border-radius:12px;
    padding:15px;
    border:1px solid #dddddd;
}

.sidebar .sidebar-content{
    background:#003366;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load Models
# ------------------------------------------------------------

@st.cache_resource
def load_models():

    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans.pkl")
    encoders = joblib.load("models/encoders.pkl")

    return scaler, kmeans, encoders


scaler, kmeans, encoders = load_models()

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

@st.cache_data
def load_data():

    file_path = "data/EduPro Online Platform.xlsx"

    users = pd.read_excel(file_path, sheet_name="Users")
    teachers = pd.read_excel(file_path, sheet_name="Teachers")
    courses = pd.read_excel(file_path, sheet_name="Courses")
    transactions = pd.read_excel(file_path, sheet_name="Transactions")

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
        suffixes=("_Course","_Teacher")
    )

    return users, teachers, courses, transactions, data


users, teachers, courses, transactions, data = load_data()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("🎓 EduPro AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "👤 Student Profile",

        "🤖 AI Recommendation",

        "📊 Analytics"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Models Loaded Successfully")

# ------------------------------------------------------------
# Dashboard Header
# ------------------------------------------------------------

st.title("🎓 EduPro AI Recommendation System")

st.write(
    "Artificial Intelligence based Course Recommendation using Machine Learning (K-Means Clustering)"
)

st.markdown("---")

# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "🏠 Dashboard":

    st.header("📊 Dashboard")

    total_students = len(users)
    total_courses = len(courses)
    total_teachers = len(teachers)
    total_transactions = len(transactions)
    total_revenue = transactions["Amount"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("👨‍🎓 Students", f"{total_students:,}")
    c2.metric("📚 Courses", f"{total_courses:,}")
    c3.metric("👨‍🏫 Teachers", f"{total_teachers:,}")
    c4.metric("💳 Transactions", f"{total_transactions:,}")
    c5.metric("💰 Revenue", f"₹{total_revenue:,.2f}")

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("Course Categories")

        category_count = (
            courses["CourseCategory"]
            .value_counts()
            .reset_index()
        )

        category_count.columns = [
            "Category",
            "Count"
        ]

        fig = px.pie(
            category_count,
            names="Category",
            values="Count",
            hole=0.45,
            title="Course Categories"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Course Levels")

        level_count = (
            courses["CourseLevel"]
            .value_counts()
            .reset_index()
        )

        level_count.columns = [
            "Level",
            "Count"
        ]

        fig2 = px.bar(
            level_count,
            x="Level",
            y="Count",
            title="Courses by Level",
            text_auto=True
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("⭐ Top Rated Courses")

        top_courses = (
            courses.sort_values(
                "CourseRating",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            top_courses[
                [
                    "CourseName",
                    "CourseCategory",
                    "CourseLevel",
                    "CourseRating"
                ]
            ],
            use_container_width=True
        )

    with col2:

        st.subheader("👨‍🏫 Top Rated Teachers")

        top_teachers = (
            teachers.sort_values(
                "TeacherRating",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            top_teachers[
                [
                    "TeacherName",
                    "Expertise",
                    "TeacherRating",
                    "YearsOfExperience"
                ]
            ],
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("💳 Payment Method Distribution")

    payment = (
        transactions["PaymentMethod"]
        .value_counts()
        .reset_index()
    )

    payment.columns = [
        "Payment",
        "Count"
    ]

    fig3 = px.bar(
        payment,
        x="Payment",
        y="Count",
        color="Payment",
        text_auto=True
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


    # ============================================================
# STUDENT PROFILE
# ============================================================

elif page == "👤 Student Profile":

    st.header("👤 Student Profile")

    user_id = st.selectbox(
        "Select Student",
        users["UserID"].sort_values().unique()
    )

    student = users[users["UserID"] == user_id].iloc[0]

    student_data = data[data["UserID"] == user_id]

    total_spent = student_data["Amount"].sum()

    avg_spent = student_data["Amount"].mean()

    courses_count = student_data["CourseID"].count()

    avg_rating = student_data["CourseRating"].mean()

    teacher_rating = student_data["TeacherRating"].mean()

    teacher_exp = student_data["YearsOfExperience"].mean()

    favorite_category = (
        student_data["CourseCategory"].mode().iloc[0]
        if not student_data.empty else "Unknown"
    )

    favorite_level = (
        student_data["CourseLevel"].mode().iloc[0]
        if not student_data.empty else "Unknown"
    )

    preferred_type = (
        student_data["CourseType"].mode().iloc[0]
        if not student_data.empty else "Unknown"
    )

    payment_method = (
        student_data["PaymentMethod"].mode().iloc[0]
        if not student_data.empty else "Unknown"
    )

    # -------------------------
    # Encode values
    # -------------------------

    gender = encoders["Gender"].transform([student["Gender"]])[0]
    fav_cat = encoders["FavoriteCategory"].transform([favorite_category])[0]
    fav_level = encoders["FavoriteLevel"].transform([favorite_level])[0]
    pref_type = encoders["PreferredType"].transform([preferred_type])[0]
    pay_method = encoders["PaymentMethod"].transform([payment_method])[0]

    feature = pd.DataFrame([{

        "Age": student["Age"],
        "Gender": gender,
        "TotalSpent": total_spent,
        "AverageSpent": avg_spent if not np.isnan(avg_spent) else 0,
        "CoursesEnrolled": courses_count,
        "AverageRating": avg_rating if not np.isnan(avg_rating) else 0,
        "FavoriteCategory": fav_cat,
        "FavoriteLevel": fav_level,
        "PreferredType": pref_type,
        "PaymentMethod": pay_method,
        "TeacherRating": teacher_rating if not np.isnan(teacher_rating) else 0,
        "TeacherExperience": teacher_exp if not np.isnan(teacher_exp) else 0

    }])

    scaled = scaler.transform(feature)

    cluster = int(kmeans.predict(scaled)[0])

    st.success(f"🎯 Predicted Cluster : {cluster}")

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Student Information")

        st.write(f"**User ID:** {user_id}")
        st.write(f"**Name:** {student['UserName']}")
        st.write(f"**Age:** {student['Age']}")
        st.write(f"**Gender:** {student['Gender']}")
        st.write(f"**Email:** {student['Email']}")

    with c2:

        st.subheader("Learning Summary")

        st.write(f"**Courses Enrolled:** {courses_count}")
        st.write(f"**Total Spent:** ₹{total_spent:.2f}")
        st.write(f"**Average Spending:** ₹{avg_spent:.2f}")
        st.write(f"**Average Rating:** {avg_rating:.2f}")
        st.write(f"**Favourite Category:** {favorite_category}")
        st.write(f"**Preferred Level:** {favorite_level}")

    st.markdown("---")

    st.subheader("Purchased Courses")

    st.dataframe(

        student_data[
            [
                "CourseName",
                "CourseCategory",
                "CourseLevel",
                "CourseRating",
                "Amount"
            ]
        ],

        use_container_width=True

    )


    # ============================================================
# AI RECOMMENDATION PAGE
# ============================================================

elif page == "🤖 AI Recommendation":

    st.header("🤖 AI Course Recommendation")

    user_id = st.selectbox(
        "Select Student",
        users["UserID"].sort_values().unique(),
        key="recommend_user"
    )

    student_data = data[data["UserID"] == user_id]

    if student_data.empty:
        st.warning("No transaction history found.")
    else:

        favorite_category = (
            student_data["CourseCategory"]
            .mode()
            .iloc[0]
        )

        purchased_courses = student_data["CourseID"].unique()

        recommendations = courses[
            (courses["CourseCategory"] == favorite_category) &
            (~courses["CourseID"].isin(purchased_courses))
        ].sort_values(
            by="CourseRating",
            ascending=False
        )

        st.success(f"Recommended based on favourite category: **{favorite_category}**")

        if recommendations.empty:
            st.info("No new recommendations available.")
        else:

            st.dataframe(
                recommendations[
                    [
                        "CourseID",
                        "CourseName",
                        "CourseCategory",
                        "CourseLevel",
                        "CourseRating",
                        "CoursePrice",
                        "CourseDuration"
                    ]
                ],
                use_container_width=True
            )

            csv = recommendations.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Recommendations",
                csv,
                file_name=f"{user_id}_recommendations.csv",
                mime="text/csv"
            )

# ============================================================
# ANALYTICS PAGE
# ============================================================

elif page == "📊 Analytics":

    st.header("📊 Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Cluster Summary")

        cluster_summary = pd.read_csv(
            "outputs/cluster_summary.csv"
        )

        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

    with col2:

        st.subheader("Student Cluster Distribution")

        segments = pd.read_csv(
            "outputs/student_segments.csv"
        )

        fig = px.histogram(
            segments,
            x="Cluster",
            color="Cluster",
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("PCA Cluster Visualization")

        if os.path.exists("outputs/pca_clusters.png"):

            st.image(
                "outputs/pca_clusters.png",
                use_container_width=True
            )

    with right:

        st.subheader("Elbow Curve")

        if os.path.exists("outputs/elbow_curve.png"):

            st.image(
                "outputs/elbow_curve.png",
                use_container_width=True
            )

    st.markdown("---")

    with open(
        "outputs/silhouette_score.txt"
    ) as file:

        score = file.read()

    st.success(score)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
<center>

### 🎓 EduPro AI Recommendation System

Developed using **Python, Streamlit, Scikit-Learn, Pandas, Plotly**

Machine Learning Algorithm:
**K-Means Clustering**

Internship Project

</center>
""",
    unsafe_allow_html=True
)