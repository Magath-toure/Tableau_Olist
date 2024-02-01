import streamlit as st
import numpy as np
from make_pred import make_prediction
import json
import pandas as pd
import plotly.express as px
import requests
import seaborn as sns
import matplotlib.pyplot as plt

# Setup data from csv
df = pd.read_csv("data/olist_order_reviews_Dataset.csv")
# sepal_length,sepal_width,petal_length,petal_width,species

# Setup title page
st.set_page_config(page_title="Prediction")
st.header("Prediction - Olist Dataset")
st.markdown("Utilize the RandomForestClassifier to make predictions for the classification of the review score."
            "The predictions will be displayed on the graphs below to intuitively understand how they were made.")
st.sidebar.header("Make Prediction")

rev_id = st.sidebar.text_input("Review id")
ord_id = st.sidebar.text_input("Order id")
rev_sco = st.sidebar.text_input("Review Score")
rev_com_tit = st.sidebar.text_input("Review Comment Title")
rev_com_mes = st.sidebar.text_input("Review Comment Message")
rev_cre_dat = st.sidebar.text_input("Review Creation Date")
rev_ans_tim = st.sidebar.text_input("Review Answer Timestamp")
make_pred_API = st.sidebar.button("Predict")

#analyse univariée
# Affichage de l'histogramme
plt.hist(df['review_score'],
         bins=20, color='skyblue', edgecolor='black')
plt.title("Review score vs Review id")
plt.xlabel('Review Score')
plt.ylabel('Review id')
#sns.scatterplot(
    #df,
   # x="review_score",
   # y="review_id",
   # title="Review score vs Review id",
    #c="review_score")

plot2 = px.scatter(
    df,
    x="review_id",
    y="order_id",
    title="Review id Vs Order id",
    color="review_score")
plot3 = px.scatter(
    df,
    x="review_creation_date",
    y="review_answer_timestamp",
    title="Review Creation date Vs Reviews answer timestamp",
    color="review_score")
plot4 = px.scatter(
    df,
    x="review_comment_message",
    y="review_comment_title",
    title="Review Comment Message Vs Review Comment Title",
    color="review_score")

# Launch prediction with API
if make_pred_API:
    # Construire l'URL avec les paramètres
    url = f"http://localhost:8000/{str(rev_id)}/{str(ord_id)}/{int(rev_sco)}/{str(rev_cre_dat)}/{str(rev_com_mes)}/{str(rev_com_tit)}/{str(rev_ans_tim)}"

    # Envoyer la requête à FastAPI
    response = requests.get(url)

    # Vérifier si la requête a réussi (statut 200)
    if response.status_code == 200:
        review_score_pred = response.json()["prediction"]
        st.success(f"Prediction result: {review_score_pred} ")
    else:
        st.error("Error in prediction request.")

    # Transformer mes x1/x2/x3/x4 en df
    p1 = [str(rev_id), str(ord_id), int(rev_sco), str(rev_cre_dat), str(rev_com_mes), str(rev_com_tit), str(rev_ans_tim)]
    x = np.array([p1])
    row = {"review_id": [str(rev_id)],
           "order_id": [str(ord_id)],
           "review_score": [int(rev_sco)],
           "review_creation_date": [str(rev_cre_dat)],
           "review_comment_message": [str(rev_com_mes)],
           "review_comment_title": [str(rev_com_tit)],
           "review_answer_timestamp": [str(rev_ans_tim)]}

    p1_df = pd.DataFrame(row)

    plt.add_scatter(x=p1_df["review_id"], 
                      y=p1_df["order_id"],
                      mode='markers',  
                      name=review_score_pred,  
                      marker=dict(
                            c='red',  # Couleur des points
                            size=30,  # Taille des points
                            symbol='circle',  # Type de marqueur (vous pouvez choisir parmi divers symboles)
                            line=dict(
                                color='white',  # Couleur de la bordure des points
                                width=10  # Largeur de la bordure des points
                            )
                      ))
    plot2.add_scatter(x=p1_df["review_id"], 
                      y=p1_df["order_id"],
                      mode='markers',  
                      name=review_score_pred,  
                      marker=dict(
                            color='red',  # Couleur des points
                            size=30,  # Taille des points
                            symbol='circle',  # Type de marqueur (vous pouvez choisir parmi divers symboles)
                            line=dict(
                                color='white',  # Couleur de la bordure des points
                                width=10  # Largeur de la bordure des points
                            )
    ))
    plot3.add_scatter(x=p1_df["review_creation_date"], 
                      y=p1_df["review_answer_timestamp"],
                      mode='markers',  
                      name=review_score_pred,  
                      marker=dict(
                            color='red',  # Couleur des points
                            size=30,  # Taille des points
                            symbol='circle',  # Type de marqueur (vous pouvez choisir parmi divers symboles)
                            line=dict(
                                color='black',  # Couleur de la bordure des points
                                width=10  # Largeur de la bordure des points
                            )))
    plot4.add_scatter(x=p1_df["review_comment_message"], 
                      y=p1_df["review_comment_title"],
                      mode='markers',  
                      name=review_score_pred,  
                      marker=dict(
                            color='red',  
                            size=30,  
                            symbol='circle',  
                            line=dict(
                                color='black', 
                                width=10 
                            )))
#st.plotly_chart(plot1)
plt.show()
st.plotly_chart(plot2)
st.plotly_chart(plot3)
st.plotly_chart(plot4)


























# Managing input data
# p1 = ["", "", "", ""]

# plot1 = px.scatter(
#     df,
#     x="petal_length",
#     y="petal_width",
#     title="Petal Length vs Petal Width",
#     color="species")

# plot2 = px.scatter(
#     df,
#     x="sepal_length",
#     y="petal_length",
#     title="Sepal Length vs Petal Length",
#     color="species")

# # Launch prediction with API
# if make_pred_API:
#     # Construire l'URL avec les paramètres
#     url = f"http://localhost:8000/{float(sep_len)}/{float(sep_wid)}/{float(pet_len)}/{float(pet_wid)}"

#     # Envoyer la requête à FastAPI
#     response = requests.get(url)

#     # Vérifier si la requête a réussi (statut 200)
#     if response.status_code == 200:
#         species_pred = response.json()["prediction"]
#         st.success(f"Prediction result: {species_pred}")
#     else:
#         st.error("Error in prediction request.")

#     p1 = [float(sep_len), float(sep_wid), float(pet_len), float(pet_wid)]
#     row = {"sepal_length": [float(sep_len)],
#            "sepal_width": [float(sep_wid)],
#            "petal_length": [float(pet_len)],
#            "petal_width": [float(pet_wid)]}
#     p1_df = pd.DataFrame(row)

#     st.subheader(f"Predicted Species: {species_pred}")
#     plot1.add_scatter(x=p1_df["petal_length"], 
#                       y=p1_df["petal_width"],
#                       mode='markers',  
#                       name=species_pred,  
#                       marker=dict(
#                             color='red',  # Couleur des points
#                             size=10,  # Taille des points
#                             symbol='circle',  # Type de marqueur (vous pouvez choisir parmi divers symboles)
#                             line=dict(
#                                 color='white',  # Couleur de la bordure des points
#                                 width=2  # Largeur de la bordure des points
#                             )
#                       ))
#     plot2.add_scatter(x=p1_df["sepal_length"], 
#                       y=p1_df["petal_length"],
#                       mode='markers',  
#                       name=species_pred,  
#                       marker=dict(
#                             color='red',  # Couleur des points
#                             size=10,  # Taille des points
#                             symbol='circle',  # Type de marqueur (vous pouvez choisir parmi divers symboles)
#                             line=dict(
#                                 color='white',  # Couleur de la bordure des points
#                                 width=2  # Largeur de la bordure des points
#                             )
#     ))

# # Making a prediction and displaying data
# if make_pred:
#     p1 = [float(sep_len), float(sep_wid), float(pet_len), float(pet_wid)]
#     x = np.array([p1])
#     row = {"sepal_length": [float(sep_len)],
#            "sepal_width": [float(sep_wid)],
#            "petal_length": [float(pet_len)],
#            "petal_width": [float(pet_wid)]}

#     p1_df = pd.DataFrame(row)
#     species_pred = make_prediction(x)

#     st.subheader(f"Predicted Species: {species_pred}")
#     plot1.add_scatter(x=p1_df["petal_length"], 
#                       y=p1_df["petal_width"],
#                       mode='markers',  
#                       name=species_pred,  
#                       marker=dict(
#                             color='red',  # Couleur des points
#                             size=10,  # Taille des points
#                             symbol='circle',  # Type de marqueur (vous pouvez choisir parmi divers symboles)
#                             line=dict(
#                                 color='white',  # Couleur de la bordure des points
#                                 width=2  # Largeur de la bordure des points
#                             )
#                       ))
#     plot2.add_scatter(x=p1_df["sepal_length"], 
#                       y=p1_df["petal_length"],
#                       mode='markers',  
#                       name=species_pred,  
#                       marker=dict(
#                             color='red',  # Couleur des points
#                             size=10,  # Taille des points
#                             symbol='circle',  # Type de marqueur (vous pouvez choisir parmi divers symboles)
#                             line=dict(
#                                 color='white',  # Couleur de la bordure des points
#                                 width=2  # Largeur de la bordure des points
#                             )
#     ))

# st.plotly_chart(plot1)
# st.plotly_chart(plot2)

# print('toto4')

# #5.2/2.7/3.9/1.4