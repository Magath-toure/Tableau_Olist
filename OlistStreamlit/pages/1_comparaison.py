import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Setup data
df = pd.read_csv("data/olist_order_reviews_Dataset.csv")
# columns name: sepal_length,sepal_width,petal_length,petal_width,species

# Make page
st.set_page_config(page_title="Olist Dataset")
st.header("Comparison - Olist Dataset")
st.markdown("Explore the variables to understand their relationships and how they correlate with the species. "
            "As patterns emerge, we can intuitively understand how the RandomForestClassifier makes decisions in classifying data.")
st.sidebar.header("Variable Comparison")

# Setting graph to display
options = st.sidebar.radio("Select compairison",
                           options=["Review Creation date Vs Reviews answer timestamp",
                                    "Review Comment Message Vs Review Comment Title",
                                    "Review score vs Review id",
                                    "Review id Vs Order id"])

if options == "Review Creation date Vs Reviews answer timestamp":
    plot = px.scatter(
        (df),
        x="review_creation_date",
        y="review_answer_timestamp",
        color="review_score",
        title=options)
    # Personnalisation des axes
    plot.update_xaxes(title_text="Review Creation Date")
    plot.update_yaxes(title_text="Review Answer Timestamp")

elif options == "Review Comment Message Vs Review Comment Title":
    plot = px.scatter(
        (df),
        x="review_comment_message",
        y="review_comment_title",
        color="review_score",
        title=options)
    # Personnalisation des axes
    plot.update_xaxes(title_text="Review Comment Message")
    plot.update_yaxes(title_text="Review Comment Tittle")

elif options == "Review score vs Review id":
    plot = px.scatter(
        (df),
        x="review_score",
        y="review_id",
        color="review_score",
        title=options)
    # Personnalisation des axes
    plot.update_xaxes(title_text="Review Score")
    plot.update_yaxes(title_text="Review id")

elif options == "Review id Vs Order id":
    plot = px.scatter(
        (df),
        x="review_id",
        y="order_id",
        color="review_score",
        title=options)
    # Personnalisation des axes
    plot.update_xaxes(title_text="Review id")
    plot.update_yaxes(title_text="Order id")

#t.plotly_chart(plot)




st.plotly_chart(plot)