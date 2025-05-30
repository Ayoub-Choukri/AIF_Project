import os 
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import requests 



PATH_MODULES = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Modules"
sys.path.append(PATH_MODULES)

PATH_MODELS = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Models"
sys.path.append(PATH_MODELS)

URL_API_WEBAPP = "127.0.0.1"
PORT_API_WEBAPP = 5000
URL_API_MODEL = "127.0.0.1"
PORT_API_MODEL = 5001

App = Flask(__name__)

Service = None

@App.route('/', methods=['GET'])
def index():
    return render_template('Index.html')

@App.route('/define_service', methods=['POST'])
def Define_Service():
    global Service
    data = request.get_json()
    Service = data['Service']
    print(f"==== Service Defined in the Web Page: {Service} ====")

    # Send the Service to the Model API
    Data_To_Send = {"Service": Service}
    try:
        response = requests.post("http://" +URL_API_MODEL + ":" + str(PORT_API_MODEL) +"/define_service", json=Data_To_Send)
        response = response.json()
        print("==== Service Defined in the Model API ====")
    except Exception as e:
        print(f"==== Error: {e} ====")
        return jsonify({"error": str(e)}), 500
    

    return jsonify({"message": "Service Fully Defined"}), 200





@App.route('/prediction_page', methods=['GET'])
def prediction_page():
    global Service 
    if Service == "Movie_Genre_Predictor":
        return redirect(url_for('Movie_Genre_Predictor'))
    elif Service == "Movie_Recommender":
        return redirect(url_for('Movie_Recommender'))
    elif Service == None:
        return jsonify({"error": "Service Not Defined"}), 400
    else:
        return jsonify({"error": "Service Not Found"}), 404
        

@App.route('/movie_genre_detector', methods=['GET'])
def Movie_Genre_Predictor():
    return render_template('Movie_Genre_Detector.html')



@App.route('/movie_recommender', methods=['GET'])
def Movie_Recommender():
    return render_template('Movie_Recommender.html')





PATH_SAVING_IMAGES = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Webapp/Saved_Images"
# Predict
@App.route('/save_image_to_predict', methods=['POST'])
def save_image_to_predict():
    """Enregistre une image envoyée en POST."""
    print("==== Saving Image To Predict ====")

    # Vérifier si un fichier est bien envoyé
    if 'file' not in request.files:
        return jsonify({"error": "No image sent"}), 400

    image = request.files['file']

    try:
        # Sauvegarde de l'image
        save_path = os.path.join(PATH_SAVING_IMAGES, "Image_To_Predict.png")
        image.save(save_path)

        print(f"==== Image Saved at {save_path} ====")
        return jsonify({"message": "Image successfully saved", "path": save_path}), 200
    
    except Exception as e:
        print(f"==== Error: {e} ====")
        return jsonify({"error": str(e)}), 500
    

@App.route('/ask_to_predict', methods=['GET'])
def ask_to_predict():
    """Demande de prédiction de l'image enregistrée."""
    global Service
    if Service == "Movie_Genre_Predictor":
        Result_Json, Result_Status = ask_to_predict_movie_genre_predictor()
        return Result_Json, Result_Status
    elif Service == "Movie_Recommender":
        Result_Json, Result_Status = ask_to_predict_movie_recommender()
        return Result_Json, Result_Status
    elif Service == None:
        return jsonify({"error": "Service Not Defined"}), 400
    else:
        return jsonify({"error": "Service Not Found"}), 404
    

    





def ask_to_predict_movie_genre_predictor():
    """Demande de prédiction de l'image enregistrée."""

    print("==== Asking to Predict Movie Genre ====")

    Data_To_Send = {
        "Image_Path": PATH_SAVING_IMAGES + "/Image_To_Predict.png"
    }

    try:
        response = requests.post("http://" +URL_API_MODEL + ":" + str(PORT_API_MODEL) +"/predict", json=Data_To_Send)
        response = response.json()
        print("==== Image Predicted ====")
        return jsonify(response), 200
    except Exception as e:
        print(f"==== Error: {e} ====")
        return jsonify({"error": str(e)}), 500
    

def ask_to_predict_movie_recommender():
    """Demande de prédiction de l'image enregistrée."""

    print("==== Asking to Predict Movie Recommender ====")

    Data_To_Send = {
        "Image_Path": PATH_SAVING_IMAGES + "/Image_To_Predict.png"
    }

    try:
        response = requests.post("http://" +URL_API_MODEL + ":" + str(PORT_API_MODEL) +"/predict", json=Data_To_Send)
        response = response.json()
        print("==== Movie Neighbors Predicted ====")
        return jsonify(response), 200
    except Exception as e:
        print(f"==== Error: {e} ====")
        return jsonify({"error": str(e)}), 500

    


# Route pour servir les images enregistrées
@App.route('/Saved_Images/<filename>', methods=['GET'])
def get_saved_images(filename):
    return send_from_directory("/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Webapp/Saved_Images", filename)


PATH_SAVING_RECOMMENDED_IMAGES = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Webapp/Recommended_Images"
@App.route('/Recommended_Images/<filename>', methods=['GET'])
def get_recommended_image(filename):
    print("==== Getting Recommended Image ====")
    return send_from_directory(PATH_SAVING_RECOMMENDED_IMAGES, filename)




# Route pour refaire une prédiction
@App.route('/predict_again', methods=['GET'])
def predict_again():
    # Aller à la route ask_to_predict
    return ask_to_predict()



if __name__ == "__main__":
    App.run(debug=True, host=URL_API_WEBAPP, port=PORT_API_WEBAPP)

