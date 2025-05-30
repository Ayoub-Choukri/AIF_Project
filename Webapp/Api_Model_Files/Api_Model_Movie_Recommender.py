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
import os
from flask import Blueprint, Flask, request, jsonify, render_template, send_from_directory



Api_Model_Movie_Recommender = Blueprint('Api_Model_Movie_Recommender', __name__,url_prefix='/Movie_Recommender')

PATH_MODELS_MOVIE_RECOMMENDER = "Models/Movie_Recommender"
PATH_SAVED_MODELS_MOVIE_RECOMMENDER = "Saved_Models/Movie_Recommender"
PATH_MODULES_MOVIE_RECOMMENDER = "Modules/Movie_Recommender"
sys.path.append(PATH_MODELS_MOVIE_RECOMMENDER)
sys.path.append(PATH_SAVED_MODELS_MOVIE_RECOMMENDER)
sys.path.append(PATH_MODULES_MOVIE_RECOMMENDER)


Name_Label_To_Index = {"action" : 0 , "animation" : 1, "comedy" : 2, "documentary" : 3, "drama" : 4 , "fantasy" : 5 , "horror" : 6  , "romance" : 7
                       ,"science Fiction" : 8 , "thriller" : 9}

Index_To_Name_Label = {v:k for k,v in Name_Label_To_Index.items() }






from Resnet_Movie_Recommender import Get_ResNet_Movie_Recommender

from Embeddings_Computing import Load_Annoy_Index
from KNN import Compute_KNN, Import_Images_From_Paths, Display_Images


# Détermine si on est dans Docker
IN_DOCKER = os.environ.get('IN_DOCKER', False)

# Configuration des URLs
URL_API_WEBAPP = "webapp" if IN_DOCKER else "127.0.0.1"
URL_API_MODEL = "model" if IN_DOCKER else "0.0.0.0"
PORT_API_WEBAPP = 5000
PORT_API_MODEL = 5001


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


Model_Loaded = False
Model = None
Annoy_Index_Loaded = False
Annoy_Index = None
Data_Paths_And_Labels_Loaded = False
Data_Paths_And_Labels = None





Embedding_Size = 2048
Metric = 'angular'





# Load the model
@Api_Model_Movie_Recommender.route('/load_model', methods=['GET'])
def Load_Model():
    global Model_Loaded
    global Model
    global Annoy_Index
    global Annoy_Index_Loaded
    global Data_Paths_And_Labels
    global Data_Paths_And_Labels_Loaded
    # Reset global variables of the model
    Model_Loaded = False
    Model = None


    Model = Get_ResNet_Movie_Recommender(Pretrained=True, ResNet_Version=50, Num_Classes=len(Name_Label_To_Index))
    Load_Annoy()
    Model = Model.to(DEVICE)
    Model_Loaded = True
    
    Load_Data_Paths_And_Labels()

    print("==== Model Loaded , Annoy Index Loaded and Data Paths and Labels Loaded ====")
    return jsonify({"message": "Model loaded and Annoy Index Loaded and Data Paths and Labels Loaded"}), 200




PATH_ANNOY_INDEX = "Data/Movie_Recommender/Annoy_Index/Annoy_Index.ann"
def Load_Annoy():
    global Annoy_Index
    global Annoy_Index_Loaded
    print("==== Loading Annoy Index ====")

    try:
        Annoy_Index = Load_Annoy_Index(PATH_ANNOY_INDEX, Embedding_Size, Metric)
        Annoy_Index_Loaded = True
        print("==== Annoy Index Loaded ====")
    except Exception as e:
        Annoy_Index_Loaded = False
        Annoy_Index = None
        print(f"==== Failed to Load Annoy Index: {str(e)} ====")




PATH_DATA_PATHS_AND_LABELS = "Data/Movie_Recommender/Data_Paths_And_Labels/Data_Paths_And_Labels.csv"
def Load_Data_Paths_And_Labels():
    global Data_Paths_And_Labels
    global Data_Paths_And_Labels_Loaded
    print("==== Loading Data Paths and Labels ====")
    Data_Paths_And_Labels = pd.read_csv(PATH_DATA_PATHS_AND_LABELS)
    Data_Paths_And_Labels_Loaded = True
    print("==== Data Paths and Labels Loaded ====")








PATH_SAVING_RECOMMENDED_IMAGES = "Webapp_2/Recommended_Images"

@Api_Model_Movie_Recommender.route('/predict', methods=['POST'])
def Predict_Movie_Recommender():
    """Prédit les films les plus proches de l'image reçue."""

    print("==== Predicting Movie Recommender ====")

    global Model_Loaded
    global Model
    global Annoy_Index
    global Annoy_Index_Loaded

    if not Model_Loaded: 
        return jsonify({"error": "Model not loaded"}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image file received"}), 400

    file = request.files['image']

    # Charger l'image
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Unable to open image: {str(e)}"}), 500

    # Prétraitement
    transform = transforms.Compose([
        transforms.Resize((280, 185)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)
    Model = Model.to(DEVICE)
    Model.eval()
    with torch.no_grad():
        Embedding = Model(image).cpu().numpy()[0]

    if not Annoy_Index_Loaded:
        return jsonify({"error": "Annoy Index not loaded"}), 400

    Neighbors_Paths = Compute_KNN(
        Query_Embedding=Embedding,
        Data_Paths_And_Labels=Data_Paths_And_Labels,
        Annoy_Index=Annoy_Index,
        Num_Neighbors=5
    )

    # _ = save_recommended_images(Neighbors_Paths)

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
