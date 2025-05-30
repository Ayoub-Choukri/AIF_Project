from flask import Blueprint, request, jsonify, render_template
import sys

import requests 
import os



PATH_MODULES = "Modules/"
sys.path.append(PATH_MODULES)

PATH_MODELS = "Models/"
sys.path.append(PATH_MODELS)


Api_Webapp_Text_Based_Movie_Recommender = Blueprint('Api_Webapp_Text_Based_Movie_Recommender', __name__,  url_prefix='/Text_Based_Movie_Recommender')


IN_DOCKER = os.environ.get('IN_DOCKER', False)


# Configuration des URLs
URL_API_WEBAPP = "0.0.0.0" if IN_DOCKER else "127.0.0.1"
URL_API_MODEL = "model" if IN_DOCKER else "0.0.0.0"
PORT_API_WEBAPP = 5000
PORT_API_MODEL = 5001


NAME_SERVICE = "Text_Based_Movie_Recommender"






@Api_Webapp_Text_Based_Movie_Recommender.route('/', methods=['GET'])
def Prediction_Page_Text_Based_Movie_Recommender():
    return render_template('Text_Based_Movie_Recommender.html')



EMBEDDING_TYPE = None
@Api_Webapp_Text_Based_Movie_Recommender.route('/set_embedding_type', methods=['POST'])
def choose_embedding_type():
    global EMBEDDING_TYPE
    # Récupérer le type d'embedding choisi par l'utilisateur
    embedding_type = request.form.get('embedding_type')

    if embedding_type == 'WORD2VEC_SPACY':
        EMBEDDING_TYPE = 'WORD2VEC_SPACY'
    elif embedding_type == 'TFIDF':
        EMBEDDING_TYPE = 'TFIDF'
    elif embedding_type == 'DistillBERT':
        EMBEDDING_TYPE = 'DistillBERT'
    else:
        EMBEDDING_TYPE = None
        return jsonify({'error': 'Invalid embedding type'}), 400


    # Envoyer l'embedding_type au modèle API
    url_model = f"http://{URL_API_MODEL}:{PORT_API_MODEL}/{NAME_SERVICE}/set_embedding_type"
    data = {'embedding_type': EMBEDDING_TYPE}

    try:
        response = requests.post(url_model, json=data)
        if response.status_code == 200:
            return jsonify({'message': 'Embedding type set successfully both in webapp and model API'}), 200
        else:
            return jsonify({'error': 'Failed to set embedding type'}), 500
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Failed to connect to model API'}), 500



@Api_Webapp_Text_Based_Movie_Recommender.route('/load_model', methods=['GET'])
def load_model():
    try:
        # Construire l'URL complète
        url_model = f"http://{URL_API_MODEL}:{PORT_API_MODEL}/{NAME_SERVICE}/load_model"

        # Appel HTTP vers l'API modèle
        response = requests.get(url_model)
        data = response.json()

        if response.status_code == 200:
            print("✅ Modèle chargé depuis l'API modèle.")
        else:
            print("⚠️ Erreur lors du chargement du modèle:", data)

    except Exception as e:
        print("❌ Exception lors de l'appel au modèle:", str(e))

    return jsonify({'message': 'Model loaded successfully'}), 200





@Api_Webapp_Text_Based_Movie_Recommender.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    movie_description = request.form.get('movie_description')

    # Vérifier si les champs sont remplis
    if not movie_description:
        return jsonify({'error': 'Movie description is required'}), 400

    # Envoyer les données au modèle API
    url_model = f"http://{URL_API_MODEL}:{PORT_API_MODEL}/{NAME_SERVICE}/predict"
    data = {
        'query': movie_description,
    }

    try:
        response = requests.post(url_model, json=data)
        if response.status_code == 200:
            recommended_movies_titles = response.json().get('recommended_movies_titles', [])
            recommended_movies_descriptions = response.json().get('recommended_movies_descriptions', [])    

        else:
            print("⚠️ Erreur lors de la prédiction:", response.json())
            return jsonify({'error': 'Failed to get predictions from model API'}), 500
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Failed to connect to model API'}), 500
    
    # Retourner les réponses au format JSON
    return jsonify({
        'recommended_movies_titles': recommended_movies_titles,
        'recommended_movies_descriptions': recommended_movies_descriptions
    }), 200


