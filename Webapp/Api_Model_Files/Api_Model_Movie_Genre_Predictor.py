import sys
import torch
from PIL import Image
from torchvision import transforms
from PIL import Image
import os
from flask import Blueprint, request, jsonify



Api_Model_Movie_Genre_Predictor = Blueprint('Api_Model_Movie_Genre_Predictor', __name__,url_prefix="/Movie_Genre_Predictor")

PATH_MODULES_MOVIE_GENRE_PREDICTOR = "Modules/Movie_Genre_Predictor"
PATH_MODELS_MOVIE_GENRE_PREDICTOR = "Models/Movie_Genre_Predictor"
sys.path.append(PATH_MODULES_MOVIE_GENRE_PREDICTOR)
sys.path.append(PATH_MODELS_MOVIE_GENRE_PREDICTOR)


from Resnet_Movie_Genre_Predictor import Get_ResNet

# Détermine si on est dans Docker
IN_DOCKER = os.environ.get('IN_DOCKER', False)

# Configuration des URLs
URL_API_WEBAPP = "webapp" if IN_DOCKER else "127.0.0.1"
URL_API_MODEL = "model" if IN_DOCKER else "0.0.0.0"
PORT_API_WEBAPP = 5000
PORT_API_MODEL = 5001

PATH_SAVED_MODELS_MOVIE_GENRE_PREDICTOR = "Saved_Models/Movie_Genre_Predictor"


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


Name_Label_To_Index = {"action" : 0 , "animation" : 1, "comedy" : 2, "documentary" : 3, "drama" : 4 , "fantasy" : 5 , "horror" : 6  , "romance" : 7
                       ,"science Fiction" : 8 , "thriller" : 9}

Index_To_Name_Label = {v:k for k,v in Name_Label_To_Index.items() }



Model_Loaded = False
Model = None


LOAD_MODEL_ARCHITECTURE_MOVIE_GENRE_PREDICTOR = False
RESNET_VERSION_MOVIE_GENRE_PREDICTOR = 18



# Load the model
@Api_Model_Movie_Genre_Predictor.route('/load_model', methods=['GET'])
def Load_Model():
    global Model_Loaded
    global Model
    # Reset global variables of the model
    Model_Loaded = False
    Model = None


    Path_Model = PATH_SAVED_MODELS_MOVIE_GENRE_PREDICTOR + "/Movie_Genre_Predictor.pth"
    PATH_MODEL_WEIGHTS = PATH_SAVED_MODELS_MOVIE_GENRE_PREDICTOR + "/Movie_Genre_Predictor_WEIGHTS.pth"
    print("==== Loading Model Weights ====")
    if not LOAD_MODEL_ARCHITECTURE_MOVIE_GENRE_PREDICTOR:
        Model = Get_ResNet(Pretrained=True, ResNet_Version=RESNET_VERSION_MOVIE_GENRE_PREDICTOR, Num_Classes=len(Name_Label_To_Index))
        Model.load_state_dict(torch.load(PATH_MODEL_WEIGHTS,weights_only=True))
    else:
        print("==== Loading Model Architechture ====")
        Model = torch.load(Path_Model)


    Model.eval()
    Model = Model.to(DEVICE)    
    Model_Loaded = True

    print("==== Model for Genre_Predicting Loaded ====")
    return jsonify({"message": "Model loaded"}), 200






@Api_Model_Movie_Genre_Predictor.route('/predict', methods=['POST'])
def Predict_Movie_Genre_Predictor():
    """Prédit la couleur de l'image reçue en binaire."""

    print("==== Predicting Image ====")

    global Model_Loaded
    global Model
    global Name_Label_To_Index
    global Index_To_Name_Label

    if not Model_Loaded: 
        return jsonify({"error": "Model not loaded"}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image file received"}), 400

    file = request.files['image']

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Unable to open image: {str(e)}"}), 500

    # Get Original Image Size
    original_size = image.size

    # Prétraitement de l'image
    transform = transforms.Compose([
        transforms.Resize((280, 185)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(DEVICE)

    Model.eval()
    with torch.no_grad():
        predictions_energy = Model(image)
        predictions_probas = torch.nn.functional.softmax(predictions_energy, dim=1)
        predictions_probas = predictions_probas.squeeze(0)

    predictions_probas = predictions_probas.cpu().numpy()
    Predictions_Dict = {
        Index_To_Name_Label[i]: round(predictions_probas[i].item() * 100, 2)
        for i in range(len(Name_Label_To_Index))
    }

    Best_Prediction = max(Predictions_Dict, key=Predictions_Dict.get)

    Send = {
        "Best_Prediction": Best_Prediction,
        "Probas": Predictions_Dict
    }

    print("==== Movie Genre Predicted ====")
    return jsonify(Send), 200




