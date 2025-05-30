# Flask
import os
from flask import Flask
import Api_Model_Movie_Genre_Predictor
import Api_Model_Movie_Recommender
import Api_Model_Text_Based_Movie_Recommender
import Api_Model_Interpretable_Movie_Genre_Predictor




# Détermine si on est dans Docker
IN_DOCKER = os.environ.get('IN_DOCKER', False)

# Configuration des URLs
URL_API_WEBAPP = "webapp" if IN_DOCKER else "127.0.0.1"
URL_API_MODEL = "model" if IN_DOCKER else "0.0.0.0"
PORT_API_WEBAPP = 5000
PORT_API_MODEL = 5001




# Définir les chemins absolus vers les dossiers 'templates' et 'static'
template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')

# Crée l'application Flask en spécifiant les chemins
App = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

App.register_blueprint(Api_Model_Movie_Genre_Predictor.Api_Model_Movie_Genre_Predictor)
App.register_blueprint(Api_Model_Movie_Recommender.Api_Model_Movie_Recommender)
App.register_blueprint(Api_Model_Text_Based_Movie_Recommender.Api_Model_Text_Based_Movie_Recommender)
App.register_blueprint(Api_Model_Interpretable_Movie_Genre_Predictor.Api_Model_Interpretable_Movie_Genre_Predictor)



if __name__ == '__main__':
    App.run(URL_API_MODEL, PORT_API_MODEL, debug=True)






    

