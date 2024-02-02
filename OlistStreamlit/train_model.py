import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json
import pickle
import numpy as np
#from train_model import make_model_save

def make_model_save():

    # Import dataframe
    olist_df = pd.read_csv("data/olist_order_reviews_Dataset.csv")
   
    # Process Data
    label_encoder = LabelEncoder()
    olist_df['review_score_encoded'] = label_encoder.fit_transform(olist_df['review_score'])

    # Save processed data to new file and json
    options_title = olist_df['review_score'].unique()
    dict_encoder = {}

    for item in options_title:
        dict_encoder[str(olist_df[olist_df['review_score'] == item].iloc[0]['review_score_encoded'])] = item

    with open('encoder.json', 'w') as write_file:
        json.dump(dict_encoder, write_file, indent=4)
        
    # Separate Target and Features : x and y datas
    y = olist_df['review_score_encoded'].copy()
    x = olist_df.drop(['review_score', 'review_score_encoded'], axis=1)

    # Separate TrainSet / TestSet
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    # Train model
    model = RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(x_train, y_train)

    # Save model
    with open('main_model.pkl', 'wb') as fichier_modele:
        pickle.dump(model, fichier_modele)

