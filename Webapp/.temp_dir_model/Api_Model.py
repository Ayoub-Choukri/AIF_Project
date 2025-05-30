# Flask
import os
from flask import Flask, request, jsonify
import sys
import torch
from PIL import Image
from torchvision import transforms
import torchvision
import numpy as np
import pandas as pd
import json
from io import BytesIO
from PIL import Image
import base64

PATH_MODULES_MOVIE_GENRE_PREDICTOR = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Modules/Movie_Genre_Predictor"
PATH_SAVED_MODELS_MOVIE_GENRE_PREDICTOR = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Saved_Models/Movie_Genre_Predictor"

PATH_MODULES_MOVIE_RECOMMENDER = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Modules/Movie_Recommender"
PATH_SAVED_MODELS_MOVIE_RECOMMENDER = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Models/Movie_Recommender/"
sys.path.append(PATH_MODULES_MOVIE_GENRE_PREDICTOR)
sys.path.append(PATH_SAVED_MODELS_MOVIE_GENRE_PREDICTOR)
sys.path.append(PATH_MODULES_MOVIE_RECOMMENDER)
sys.path.append(PATH_SAVED_MODELS_MOVIE_RECOMMENDER)

from Resnet import Get_ResNet_Movie_Recommender
from Embeddings_Computing import Load_Annoy_Index
from KNN import Compute_KNN, Import_Images_From_Paths, Display_Images





URL_API_WEBAPP = "127.0.0.1"
PORT_API_WEBAPP = 5000
URL_API_MODEL = "127.0.0.1"
PORT_API_MODEL = 5001

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


Name_Label_To_Index = {"action" : 0 , "animation" : 1, "comedy" : 2, "documentary" : 3, "drama" : 4 , "fantasy" : 5 , "horror" : 6  , "romance" : 7
                       ,"science Fiction" : 8 , "thriller" : 9}

Index_To_Name_Label = {v:k for k,v in Name_Label_To_Index.items() }


Embedding_Size = 2048
Metric = 'angular'

App = Flask(__name__)

Model_Loaded = False
Model = None
Annoy_Index_Loaded = False
Annoy_Index = None
Data_Paths_And_Labels_Loaded = False
Data_Paths_And_Labels = None

Service = None

@App.route('/define_service', methods=['POST'])
def Define_Service():
    global Service

    try : 
        data = request.get_json()
        Service = data['Service']


    
        # Try to load the model
        try:
            Result = Load_Model()
            print(f"==== Service Defined: {Service} and Model Loaded ====")
            return Result
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        

    except Exception as e:
        return jsonify({"error": str(e)}), 400






# Load the model
@App.route('/load_model', methods=['GET'])
def Load_Model():
    global Model_Loaded
    global Model
    global Service
    print(f"==== Service: {Service} ====")
    print("==== Loading Model ====")
    if Service == "Movie_Genre_Predictor":
        Path_Model = PATH_SAVED_MODELS_MOVIE_GENRE_PREDICTOR + "/Movie_Genre_Predictor.pth"

        Model = torch.load(Path_Model)
        Model = Model.to(DEVICE)    
        Model_Loaded = True
        return jsonify({"message": "Model loaded"}), 200
    elif Service == "Movie_Recommender":
        Model = Get_ResNet_Movie_Recommender(Pretrained=True, ResNet_Version=50, Num_Classes=len(Name_Label_To_Index))
        Load_Annoy()
        Model = Model.to(DEVICE)
        Model_Loaded = True
        
        Load_Data_Paths_And_Labels()

        print("==== Model Loaded , Annoy Index Loaded and Data Paths and Labels Loaded ====")
        return jsonify({"message": "Model loaded and Annoy Index Loaded and Data Paths and Labels Loaded"}), 200
    else:
        return jsonify({"error": "Service not found"}), 400
    




    




PATH_SAVED_IMAGES = "/home/ayoubchoukri/Etudes/Personel/Projects/Projects_Computer/Colorizer/Webapp/Saved_Images"

@App.route('/predict', methods=['POST'])
def predict():
    global Service

    if Service == "Movie_Genre_Predictor":
        return Predict_Movie_Genre_Predictor()
    elif Service == "Movie_Recommender":
        return Predict_Movie_Recommender()
    elif Service == None : 
        return jsonify({"error": "Service not defined"}), 400
    else:
        return jsonify({"error": "Service not found"}), 400
    


def Predict_Movie_Genre_Predictor():
    """Prédit la couleur de l'image enregistrée."""

    print("==== Predicting Image ====")

    global Model_Loaded
    global Model
    global Name_Label_To_Index
    global Index_To_Name_Label

    if not Model_Loaded: 
        return jsonify({"error": "Model not loaded"}), 400

    # Récupérer le chemin de l'image depuis la requête
    data = request.get_json()
    if 'Image_Path' not in data:
        return jsonify({"error": "No image path provided"}), 400

    image_path = data['Image_Path']

    # Charger l'image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Unable to open image: {str(e)}"}), 500

    # Get Original Image Size
    original_size = image.size
    # Prétraitement de l'image
    transform = transforms.Compose([
        transforms.Resize((280, 185)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    image = image.to(DEVICE)


    Model.eval()
    # Prédiction
    with torch.no_grad():
        predictions_energy = Model(image)
        predictions_probas = torch.nn.functional.softmax(predictions_energy, dim=1)
        predictions_probas = predictions_probas.squeeze(0)

    predictions_probas = predictions_probas.cpu().numpy()
    Predictions_Dict = {Index_To_Name_Label[i] : np.round(predictions_probas[i].item()*100,2)for i in range(len(Name_Label_To_Index))}
    
    Best_Prediction = max(Predictions_Dict, key=Predictions_Dict.get)

    Send = {
        "Best_Prediction": Best_Prediction,
        "Probas": Predictions_Dict
    }


    print("==== Moovie Genre Predicted ====")
    return jsonify(Send), 200



PATH_ANNOY_INDEX = "Data/Movie_Recommender/Annoy_Index/Annoy_Index.ann"
def Load_Annoy():
    global Annoy_Index
    global Annoy_Index_Loaded
    print("==== Loading Annoy Index ====")
    Annoy_Index = Load_Annoy_Index(PATH_ANNOY_INDEX, Embedding_Size, Metric)
    Annoy_Index_Loaded = True
    print("==== Annoy Index Loaded ====")



PATH_DATA_PATHS_AND_LABELS = "Data/Movie_Recommender/Data_Paths_And_Labels/Data_Paths_And_Labels.csv"
def Load_Data_Paths_And_Labels():
    global Data_Paths_And_Labels
    global Data_Paths_And_Labels_Loaded
    print("==== Loading Data Paths and Labels ====")
    Data_Paths_And_Labels = pd.read_csv(PATH_DATA_PATHS_AND_LABELS)
    Data_Paths_And_Labels_Loaded = True
    print("==== Data Paths and Labels Loaded ====")






PATH_SAVING_RECOMMENDED_IMAGES = "Webapp/Recommended_Images"
def Predict_Movie_Recommender():
    """Prédit les films les plus proches de l'image enregistrée."""

    print("==== Predicting Movie Recommender ====")

    global Model_Loaded
    global Model
    global Annoy_Index
    global Annoy_Index_Loaded

    if not Model_Loaded: 
        return jsonify({"error": "Model not loaded"}), 400

    # Récupérer le chemin de l'image depuis la requête
    data = request.get_json()
    if 'Image_Path' not in data:
        return jsonify({"error": "No image path provided"}), 400

    image_path = data['Image_Path']

    # Charger l'image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Unable to open image: {str(e)}"}), 500


    # Prétraitement de l'image
    transform = transforms.Compose([
        transforms.Resize((280, 185)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)
    image = image.to(DEVICE)

    Model.eval()
    # Prédiction
    with torch.no_grad():
        Embedding = Model(image)
        Embedding = Embedding.cpu().numpy()
        Embedding = Embedding[0]

    # Load Annoy Index
    if Annoy_Index_Loaded == False:
        return jsonify({"error": "Annoy Index not loaded"}), 400

    # Compute KNN
    Neighbors_Paths = Compute_KNN(Query_Embedding=Embedding, Data_Paths_And_Labels=Data_Paths_And_Labels, Annoy_Index=Annoy_Index, Num_Neighbors=5)

    # Serialize Image Paths


    
    # print("==== Movie Neighbors Predicted ====")

    # return jsonify({"Neighbors_Paths": Neighbors_Paths}), 200

        # Enregistrer les images dans "Recommended_Images"
    
    _ = save_recommended_images(Neighbors_Paths)


    image_bytecodes = []
    for path in Neighbors_Paths:
        image_bytecode = convert_image_to_bytecode(path)
        if image_bytecode:
            image_bytecodes.append(image_bytecode)

    print("==== Movie Neighbors Predicted ====")

    return jsonify({"Neighbors_Images": image_bytecodes}), 200


def save_recommended_images(Neighbors_Paths):
    """Sauvegarde les images recommandées et retourne leurs chemins d'accès."""
    saved_image_paths = []

    for i, neighbor_path in enumerate(Neighbors_Paths):
        try:
            neighbor_img = Image.open(neighbor_path).convert("RGB")
            save_path = os.path.join(PATH_SAVING_RECOMMENDED_IMAGES, f"Img{i+1}.png")
            neighbor_img.save(save_path)
            print(f"Image {i+1} saved at {save_path}")

            # Ajouter le chemin accessible par l'API
            saved_image_paths.append(f"/Recommended_Images/Img{i+1}.png")

        except Exception as e:
            print(f"Error saving image {neighbor_path}: {e}")

    return saved_image_paths


def convert_image_to_bytecode(image_path):
    """Convertit une image en bytecode pour l'envoyer via l'API."""
    try:
        with Image.open(image_path) as img:
            # Convertir l'image en bytecode (format PNG)
            byte_io = BytesIO()
            img.save(byte_io, format='PNG')
            byte_io.seek(0)  # Rewind to the beginning of the byte stream
            byte_data = byte_io.read()
            return base64.b64encode(byte_data).decode('utf-8')  # Retourner le bytecode sous forme base64
    except Exception as e:
        print(f"Error converting image {image_path} to bytecode: {e}")
        return None
if __name__ == '__main__':
    App.run(URL_API_MODEL, PORT_API_MODEL, debug=True)






    

