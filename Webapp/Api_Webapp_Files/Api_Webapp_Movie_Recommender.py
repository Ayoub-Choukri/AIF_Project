from flask import Blueprint, request, jsonify, render_template
import sys
import requests 
import os



PATH_MODULES = "Modules/"
sys.path.append(PATH_MODULES)

PATH_MODELS = "Models/"
sys.path.append(PATH_MODELS)


Api_Webapp_Movie_Recommender = Blueprint('Api_Webapp_Movie_Recommender', __name__,  url_prefix='/Movie_Recommender')


IN_DOCKER = os.environ.get('IN_DOCKER', False)


# Configuration des URLs
URL_API_WEBAPP = "0.0.0.0" if IN_DOCKER else "127.0.0.1"
URL_API_MODEL = "model" if IN_DOCKER else "0.0.0.0"
PORT_API_WEBAPP = 5000
PORT_API_MODEL = 5001


NAME_SERVICE = "Movie_Recommender"




import requests

@Api_Webapp_Movie_Recommender.route('/', methods=['GET'])
def Prediction_Page_Movie_Recommender():
    try:
        # Construire l'URL de l'API modèle
        url_model = f"http://{URL_API_MODEL}:{PORT_API_MODEL}/Movie_Recommender/load_model"

        # Appeler l'API modèle
        response = requests.get(url_model)
        data = response.json()

        if response.status_code == 200:
            print("✅ Modèle Movie Recommender chargé.")
        else:
            print(f"⚠️ Erreur chargement modèle: {data}")
    except Exception as e:
        print("❌ Exception lors de l'appel au modèle Movie Recommender:", str(e))

    # Rendre la page web
    return render_template('Movie_Recommender.html')



PATH_SAVING_IMAGES = "Webapp/Saved_Images"
# Predict
@Api_Webapp_Movie_Recommender.route('/save_image_to_predict', methods=['POST'])
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
    





@Api_Webapp_Movie_Recommender.route('/ask_to_predict', methods=['GET'])
def ask_to_predict_movie_recommender():
    """Demande de prédiction avec envoi direct de l'image."""

    print("==== Asking to Predict Movie Recommender ====")

    image_path = PATH_SAVING_IMAGES + "/Image_To_Predict.png"

    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': ('Image_To_Predict.png', img_file, 'image/png')}
            response = requests.post(
                f"http://{URL_API_MODEL}:{PORT_API_MODEL}/{NAME_SERVICE}/predict",
                files=files
            )
        response = response.json()
        print("==== Movie Neighbors Predicted ====")
        return jsonify(response), 200
    except Exception as e:
        print(f"==== Error: {e} ====")
        return jsonify({"error": str(e)}), 500



