import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import os
from flask import Blueprint, request, jsonify



Api_Model_Interpretable_Movie_Genre_Predictor = Blueprint('Api_Model_Interpretable_Movie_Genre_Predictor', __name__,url_prefix="/Interpretable_Movie_Genre_Predictor")


PATH_MODULES_MOVIE_GENRE_PREDICTOR = "Modules/Movie_Genre_Predictor"
PATH_MODELS_MOVIE_GENRE_PREDICTOR = "Models/Movie_Genre_Predictor"
PATH_MODULES_INTERPRETABLE_MOVIE_GENRE_PREDICTOR = "Modules/Interpertable_Movie_Genre_Predictior"
PATH_SAVED_MODELS_MOVIE_GENRE_PREDICTOR = "Saved_Models/Movie_Genre_Predictor"

sys.path.append(PATH_MODELS_MOVIE_GENRE_PREDICTOR)
sys.path.append(PATH_MODULES_MOVIE_GENRE_PREDICTOR)
sys.path.append(PATH_MODULES_INTERPRETABLE_MOVIE_GENRE_PREDICTOR)
sys.path.append(PATH_SAVED_MODELS_MOVIE_GENRE_PREDICTOR)

from Resnet_Movie_Genre_Predictor import Get_ResNet
from LIME import Get_Explanation, Get_Image_And_Mask, predict_fn, Preprocess_Image_Lime
from Grad_Cam import Get_CAM, Get_CAM_Fused_Image, Preprocess_Image_Grad_Cam
from Gradients import Preprocess_Image_Gradients, Get_Gradients, Get_Image_Fused_With_Gradients, Visualize_Gradients
# Détermine si on est dans Docker
IN_DOCKER = os.environ.get('IN_DOCKER', False)

# Configuration des URLs
URL_API_WEBAPP = "webapp" if IN_DOCKER else "127.0.0.1"
URL_API_MODEL = "model" if IN_DOCKER else "0.0.0.0"
PORT_API_WEBAPP = 5000
PORT_API_MODEL = 5001



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


Name_Label_To_Index = {"action" : 0 , "animation" : 1, "comedy" : 2, "documentary" : 3, "drama" : 4 , "fantasy" : 5 , "horror" : 6  , "romance" : 7
                       ,"science Fiction" : 8 , "thriller" : 9}

Index_To_Name_Label = {v:k for k,v in Name_Label_To_Index.items() }



Model_Loaded = False
Model = None
INTERPRETATION_TECHNIQUE = None



LOAD_MODEL_ARCHITECTURE_MOVIE_GENRE_PREDICTOR = False
RESNET_VERSION_MOVIE_GENRE_PREDICTOR = 18






# Load the model
@Api_Model_Interpretable_Movie_Genre_Predictor.route('/load_model', methods=['GET'])
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





@Api_Model_Interpretable_Movie_Genre_Predictor.route('/predict', methods=['POST'])
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




@Api_Model_Interpretable_Movie_Genre_Predictor.route('/set_interpretation_technique', methods=['POST'])
def set_interpretation_technique():
    """Set the interpretation technique to be used."""
    global INTERPRETATION_TECHNIQUE

    technique = request.json.get('technique')

    print(f"==== Setting Interpretation Technique to {technique} Must be one of: ['Grad-CAM', 'LIME', 'Gradients'] ====")
    if technique not in ["Grad-CAM", "LIME", "Gradients"]:
        print("==== Unsupported Interpretation Technique in API Model ====")
        return jsonify({'error': 'Unsupported interpretation technique'}), 400

    INTERPRETATION_TECHNIQUE = technique

    print(f"==== Interpretation Technique Set in API MODEL to  {INTERPRETATION_TECHNIQUE} ====")
    return jsonify({'message': f'Interpretation technique set to {INTERPRETATION_TECHNIQUE}'}), 200





@Api_Model_Interpretable_Movie_Genre_Predictor.route('/interpret', methods=['POST'])
def interpret():

    global Model_Loaded
    global Model

    if not Model_Loaded or Model is None:
        print("==== Model not loaded ====")
        return jsonify({"error": "Model not loaded"}), 400
    

    if 'image' not in request.files:
        return jsonify({"error": "No image file received"}), 400
    

    file = request.files['image']

    try:
        image_pil = Image.open(file.stream).convert("RGB")
    except Exception as e:

        return jsonify({"error": f"Unable to open image: {str(e)}"}), 500
    


    if INTERPRETATION_TECHNIQUE == "Grad-CAM":
        print("==== Interpreting Image with Grad-CAM ====")
        return interpret_image_with_grad_cam(image_pil)
    
    elif INTERPRETATION_TECHNIQUE == "LIME":
        print("==== Interpreting Image with LIME ====")
        return interpret_image_with_lime(image_pil)
    elif INTERPRETATION_TECHNIQUE == "Gradients":
        print("==== Interpreting Image with Gradients ====")
        return interpret_image_with__gradients(image_pil)
    else:
        print("==== Unsupported Interpretation Technique ====")
        return jsonify({"error": "Unsupported interpretation technique"}), 400
    

    







def interpret_image_with_lime(image_pil):
    # Prétraitement de l'image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_np = Preprocess_Image_Lime(image_pil)
    Explanation = Get_Explanation(
        Image=image_np,
        Classifier_Function=lambda x: predict_fn(x, Model, device),
        Top_Labels=3,
        Hide_Color=0,
        Num_Samples=1000
    )

    Label = Explanation.top_labels[0]  # Prendre le label le plus probable

    temp, mask, Image_With_Boundaries = Get_Image_And_Mask(
        explanation=Explanation,
        label=Label,  # Choisir le label pour lequel on veut l'explication
        positive_only=False,
        hide_rest=False,
        num_features=15,
        min_weight=0.05
    )

    # Convertir l'image PIL en base64 pour l'envoyer au front-end
    buffered = BytesIO()
    Image_With_Boundaries_pil = Image.fromarray((Image_With_Boundaries * 255).astype(np.uint8))
    Image_With_Boundaries_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_data = f"data:image/png;base64,{img_str}"
    print("==== Image Interpreted ====")

    return jsonify({
        "interpreted_image": img_data,
    }), 200



def interpret_image_with_grad_cam(Image_PIL):

    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Image_Tensor = Preprocess_Image_Grad_Cam(Image_PIL).to(Device)

    Target_Layer = "layer4.1.conv2"
    Cam = Get_CAM(Model, Image_Tensor, Target_Layer)

    # # Sauvegarde ou usage de l’image fusionnée
    Fused_Image = Get_CAM_Fused_Image(Image_Tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy(), Cam)

    # Convertir l'image PIL en base64 pour l'envoyer au front-end
    buffered = BytesIO()
    Fused_Image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_data = f"data:image/png;base64,{img_str}"
    print("==== Image Interpreted with Grad-CAM ====")



    # Recharger le modèle 

    Load_Model()

    return jsonify({
        "interpreted_image": img_data,
    }), 200



def interpret_image_with__gradients(image_pil):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    global Model_Loaded
    global Model

    if not Model_Loaded or Model is None:
        print("==== Model not loaded ====")
        return jsonify({"error": "Model not loaded"}), 400
    


    # Prétraitement
    image_tensor = Preprocess_Image_Gradients(image_pil).to(device)

    gradients, predicted_label = Get_Gradients(Model, image_tensor, device)

    # Visualisation
    # Visualize_Gradients(image_pil, gradients)


    Image_Fused = Get_Image_Fused_With_Gradients(image_tensor.cpu().detach().squeeze(0), gradients, alpha=0.5)

    # Convertir l'image PIL en base64 pour l'envoyer au front-end
    buffered = BytesIO()
    Image_Fused.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_data = f"data:image/png;base64,{img_str}"
    print("==== Image Interpreted with Gradients ====")
    return jsonify({
        "interpreted_image": img_data,
    }), 200





























    










