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
options = st.sidebar.radio("Select values",
                           options=["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message", "review_creation_date", "review_answer_timestamp"])
show_df = df.filter(items=[options, "review_score"])
# histogramme de la valeur numerique
# analyse univariée: histogrammes pour le svaleurs numeriques

plt.hist(['review_score'], color='skyblue', edgecolor='black')
plt.xlabel('valeures numeriques')
plt.ylabel('frequence')
plt.title('Histogramme de la colonne numerique')
plt.legend()
#plt.show()



plot1 = px.histogram(
    show_df,
    x=show_df[options],
    title=f"{options} Histogram",
    #nbins=30,
    color="review_score")
plt.show()
st.plotly_chart(plot1)

