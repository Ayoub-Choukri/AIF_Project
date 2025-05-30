import os 
from flask import Flask, render_template, send_from_directory
import Api_Webapp_Movie_Genre_Predictor
import Api_Webapp_Movie_Recommender
import Api_Webapp_Text_Based_Movie_Recommender
import Api_Webapp_Interpretable_Movie_Genre_Predictor





IN_DOCKER = os.environ.get('IN_DOCKER', False)

# Configuration des URLs
URL_API_WEBAPP = "0.0.0.0" if IN_DOCKER else "127.0.0.1"
URL_API_MODEL = "model" if IN_DOCKER else "0.0.0.0"
PORT_API_WEBAPP = 5000
PORT_API_MODEL = 5001

# Définir les chemins absolus vers les dossiers 'templates' et 'static'
template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')

# Crée l'application Flask en spécifiant les chemins
App = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

App.register_blueprint(Api_Webapp_Movie_Genre_Predictor.Api_Webapp_Movie_Genre_Predictor)
App.register_blueprint(Api_Webapp_Movie_Recommender.Api_Webapp_Movie_Recommender)
App.register_blueprint(Api_Webapp_Text_Based_Movie_Recommender.Api_Webapp_Text_Based_Movie_Recommender)
App.register_blueprint(Api_Webapp_Interpretable_Movie_Genre_Predictor.Api_Webapp_Interpretable_Movie_Genre_Predictor)



@App.route('/', methods=['GET'])
def index():
    return render_template('Index.html')



# Route pour servir les images enregistrées
@App.route('/Saved_Images/<filename>', methods=['GET'])
def get_saved_images(filename):
    return send_from_directory("Webapp_2/Saved_Images", filename)


PATH_SAVING_RECOMMENDED_IMAGES = "Webapp_2/Recommended_Images"
@App.route('/Recommended_Images/<filename>', methods=['GET'])
def get_recommended_image(filename):
    print("==== Getting Recommended Image ====")
    return send_from_directory(PATH_SAVING_RECOMMENDED_IMAGES, filename)


if __name__ == "__main__":
    App.run(debug=True, host=URL_API_WEBAPP, port=PORT_API_WEBAPP)

