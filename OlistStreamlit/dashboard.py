import streamlit as st
import pandas as pd

# Setup data
df = pd.read_csv("data/olist_order_reviews_Dataset.csv")

# Make page
st.set_page_config(page_title="Olist Dataset")
st.header("Iris Machine Learning Project")
st.markdown("Deployment of the Olist Dataset machine learning model using RandomForestClassifier.")
st.markdown("Use this dashboard to understand the data and to make predictions.")
st.markdown("")
st.image("img1.png")