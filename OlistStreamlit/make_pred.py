import json
import pickle

def make_prediction(x):

    # Charger le modèle à partir du fichier Pickle
    with open('main_model.pkl', 'rb') as fichier_modele:
        loaded_model = pickle.load(fichier_modele)

    # Faire la prédiction
    predictions_out = loaded_model.predict(x)

    # print('prediction:', predictions_out)

    # Charger le fichier encoder pour traduire la prédiction
    with open('encoder.json') as json_file:
        data = json.load(json_file)

    # Conversion prédiction brute --> traduite
    predictions_string = data[str(int(predictions_out))]

    # Retourne la valeur
    return predictions_string























# 
#     # Load the saved model
#     # loaded_model = XGBClassifier()
#     # loaded_model.load_model("main_model.model")

#     print('RandomForest')


#     

#        

#     print('predictions', predictions_out)
#     

#     # print('dict_out', dict_out)

#     

#     # print('encoder', data['0'])

#     # Returns the actual species name
#     return data[str(int(predictions_out))]
#     # return data[str(int(dict_out[0]))]