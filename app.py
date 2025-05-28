# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from PIL import Image

st.set_page_config(page_title="Non-Central Revenue Dashboard", layout="wide")
st.title("🏙️ Non-Central Revenue Performance Dashboard")
st.markdown("Analyze revenue potential and identify high-performing areas in non-central regions.")

df = pd.read_csv("unsupervised(elian).csv")

le_room = joblib.load("le_room_nocluster.pkl")
le_region = joblib.load("le_region_nocluster.pkl")
model = joblib.load("revenue_classifier_nocluster.pkl")
kmeans = joblib.load("kmeans.pkl")

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

#clusterPCA
st.subheader("🧩 Cluster Visualization (PCA)")
image = Image.open("cluster_pca.png")
st.image(image, caption="PCA Projection of KMeans Clusters", width=600)

# 🔮 Predict Revenue Category from User Input (Model-based, No-Cluster)
st.subheader("🔍 Predict Revenue Category from Input")

region_input = st.selectbox("Select a Region", sorted(df["neighbourhood_group_cleansed"].unique()))
room_input = st.selectbox("Select a Room Type", sorted(df["room_type"].unique()))
price_input = st.number_input("Enter Price per Night (in SGD s$)", min_value=0)
availability_input = st.number_input("How many days in a year this property is open for bookings (0–365)", min_value=0, max_value=365)
rating_input = st.slider("Review Score Rating", min_value=0.0, max_value=5.0, step=0.1)

# Encode inputs
region_encoded = le_region.transform([region_input])[0]
room_encoded = le_room.transform([room_input])[0]

#cluster input
# cluster_input = [[price_input, availability_input, room_encoded, region_encoded]]
# cluster_label = kmeans.predict(cluster_input)[0]

# Predict category
input_data = [[price_input, availability_input, rating_input, room_encoded, region_encoded]]
predicted_category = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)[0]

# Display results
st.markdown(f"### 🏷️ Predicted Revenue Category: `{predicted_category.upper()}`")
st.write("🔢 Prediction Confidence:")
def highlight_confidence(val):
    if val >= 0.5:
        return "background-color: #00FF00; color: black;"  # green
    elif val >= 0.33:
        return "background-color: #FFFF00; color: black;"  # yellow
    else:
        return "background-color: #FF0000; color: black;"  # red

proba_df = pd.DataFrame([prediction_proba], columns=model.classes_)
styled_df = proba_df.style.applymap(highlight_confidence)
st.dataframe(styled_df)

# Show data
with st.expander("📄 View Raw Data"):
    st.dataframe(df[["neighbourhood_group_cleansed", "price", "availability_365", "estimated_revenue", "revenue_category"]])
