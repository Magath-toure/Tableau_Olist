import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns



# Setup data
df = pd.read_csv("data/olist_order_reviews_Dataset.csv")
# Make page
st.set_page_config(page_title="Olist Dataset")
st.header("Values - Olist Dataset")
st.markdown("Explore the relationship between each individual variable and each columns. "
            "We can intuit patterns within the individual values and gain an understanding of how the data is utilized for classification.")
st.sidebar.header("Individual Values")

# Setting graph to display
plot1  = px.scatter(
        df, 
        x=df['review_id'],
        y =df['order_id'],
        color = 'review_score',
        title='Scatter Review id vs Order id'
        )


plot2  = px.histogram(
        df, 
        x=df['review_score'],
        y =df['review_id'],
        color = 'review_score',
        title='Histogramme des variables Review Score vs Review id'
        )
plot3 = px.scatter(
    df,
     x=df['review_creation_date'],
        y =df['review_answer_timestamp'],
        color = 'review_score',
        title='Scatter de Review creation date vs Review answer timestamp'

)
plot4 = px.scatter(
    df,
     x=df['review_comment_message'],
        y =df['review_comment_title'],
        color = 'review_score',
        title='Scatter de Review comment message vs Review comment title'
)

st.plotly_chart(plot1)
st.plotly_chart(plot2)
st.plotly_chart(plot3)
st.plotly_chart(plot4)




