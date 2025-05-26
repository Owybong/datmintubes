# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Non-Central Revenue Dashboard", layout="wide")
st.title("🏙️ Non-Central Revenue Performance Dashboard")
st.markdown("Analyze revenue potential and identify high-performing areas in non-central regions.")

# 📂 Load your static dataset here
df = pd.read_csv("unsupervised(elian).csv")

# Load encoders and model
le_room = joblib.load("le_room.pkl")
le_region = joblib.load("le_region.pkl")
model = joblib.load("revenue_classifier.pkl")

# Preprocess
df["room_type_encoded"] = le_room.transform(df["room_type"])
df["neighbourhood_group_cleansed_encoded"] = le_region.transform(df["neighbourhood_group_cleansed"])
df["estimated_revenue"] = df["price"] * df["availability_365"]
df["revenue_category"] = pd.qcut(df["estimated_revenue"], q=3, labels=["low", "mid", "high"])

# KPI section
total_revenue = df["estimated_revenue"].sum()
high_perf_count = (df["revenue_category"] == "high").sum()

col1, col2 = st.columns(2)
col1.metric("💰 Total Revenue Potential", f"${total_revenue:,.0f}")
col2.metric("🌟 High-Performing Listings", f"{high_perf_count}")

st.markdown("---")

# Top high-performing areas
st.subheader("📍 Top Performing Areas (High Revenue Listings)")
top_areas = df[df["revenue_category"] == "high"]["neighbourhood_group_cleansed"].value_counts().head(5)
st.bar_chart(top_areas)

# Revenue by area
st.subheader("📊 Revenue Distribution by Area")
revenue_by_area = df.groupby("neighbourhood_group_cleansed")["estimated_revenue"].sum().sort_values(ascending=False)
fig = px.bar(revenue_by_area, title="Estimated Revenue by Area", labels={"value": "Total Revenue", "index": "Area"})
st.plotly_chart(fig, use_container_width=True)

# Show data
with st.expander("📄 View Raw Data"):
    st.dataframe(df[["neighbourhood_group_cleansed", "price", "availability_365", "estimated_revenue", "revenue_category"]])
