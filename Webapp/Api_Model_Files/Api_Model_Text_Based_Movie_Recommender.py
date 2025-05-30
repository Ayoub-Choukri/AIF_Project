import sys
import pandas as pd
from flask import Blueprint, request, jsonify
from annoy import AnnoyIndex


Api_Model_Text_Based_Movie_Recommender = Blueprint('Api_Model_Text_Based_Movie_Recommender', __name__,url_prefix='/Text_Based_Movie_Recommender')

PATH_MODELS_Text_Based_Movie_Recommender = "Models/Text_Based_Movie_Recommender"
PATH_SAVED_MODELS_Text_Based_Movie_Recommender = "Saved_Models/Text_Based_Movie_Recommender"
PATH_MODULES_Text_Based_Movie_Recommender = "Modules/Text_Based_Movie_Recommender"


sys.path.append(PATH_MODELS_Text_Based_Movie_Recommender)
sys.path.append(PATH_SAVED_MODELS_Text_Based_Movie_Recommender)
sys.path.append(PATH_MODULES_Text_Based_Movie_Recommender)


from Recommendation_TFIDF import *
from Recommendation_DistillBERT import *
from Recommendation_Spacy import *

Name_Label_To_Index = {"action" : 0 , "animation" : 1, "comedy" : 2, "documentary" : 3, "drama" : 4 , "fantasy" : 5 , "horror" : 6  , "romance" : 7
                       ,"science Fiction" : 8 , "thriller" : 9}

Index_To_Name_Label = {v:k for k,v in Name_Label_To_Index.items() }




Model = None
EMBEDDING_TYPE = None
Title_Overiview_Dataset = None
Annoy_Index = None

@Api_Model_Text_Based_Movie_Recommender.route('/set_embedding_type', methods=['POST'])
def set_embedding_type():
    global EMBEDDING_TYPE
    data = request.get_json()
    EMBEDDING_TYPE = data['embedding_type']
    return jsonify({"message": "Embedding type set to {}".format(EMBEDDING_TYPE)}), 200



PATH_ANNOY_INDEX_TFIDF = "Data/Text_Based_Movie_Recommender/Annoy_Indexes/Tfidf/Annoy_Index.ann"
PATH_ANNOY_INDEX_DISTILBERT = "Data/Text_Based_Movie_Recommender/Annoy_Indexes/DistillBERT/Annoy_Index.ann"
PATH_ANNOY_INDEX_SPACY = "Data/Text_Based_Movie_Recommender/Annoy_Indexes/Spacy/Annoy_Index.ann"


PATH_DATASET = "Data/Text_Based_Movie_Recommender/Title_Overview_Dataset.csv"

@Api_Model_Text_Based_Movie_Recommender.route('/load_model', methods=['GET'])
def load_model():
    global Model
    global Annoy_Index
    global Title_Overiview_Dataset

    # Load the model
    load_model_response = Load_Model()

    if load_model_response[1] != 200:
        return load_model_response
    else:
        print("Model loaded successfully")

    # Load the Annoy index
    load_annoy_index_response = Load_Annoy_Index()
    if load_annoy_index_response[1] != 200:
        return load_annoy_index_response
    else:
        print("Annoy index loaded successfully")

    # Load the dataset
    load_dataset_response = Load_Title_Overview_Dataset()
    if load_dataset_response[1] != 200:
        return load_dataset_response
    else:
        print("Dataset loaded successfully")

    return jsonify({"message": "Model, Annoy index, and dataset loaded successfully"}), 200



def Load_Model():
    global Model
    global EMBEDDING_TYPE

    if EMBEDDING_TYPE == "TFIDF":
        print("Loading TFIDF model...")
        Response = Load_Model_TFIDF()
        return Response
    elif EMBEDDING_TYPE == "DistillBERT":
        print("Loading DistillBERT model...")
        Response = Load_Model_DistilBERT()
        print("DistillBERT model loaded successfully")
        return Response
    elif EMBEDDING_TYPE == "WORD2VEC_SPACY":
        print("Loading Spacy model...")
        Response = Load_Spacy_Model()
        return Response
    else:
        return jsonify({"error": f"Invalid embedding type. Expected 'TFIDF', 'DistillBERT', or 'WORD2VEC_SPACY', but got '{EMBEDDING_TYPE}'."}), 400




def Load_Annoy_Index():
    global Annoy_Index
    global EMBEDDING_TYPE

    if EMBEDDING_TYPE == "TFIDF":
        Dimensions_Path = "Data/Text_Based_Movie_Recommender/Embeddings/TFIDF/Embeddings_Title_Overview_Dataset_TFIDF_dimensions.csv"
        TFIDF_Vectorizer_Path = "Saved_Models/Text_Based_Movie_Recommender/TFIDF/TFIDF_Vectorizer.pkl"
        # Load the dimensions of the TF-IDF matrix
        Dimensions = pd.read_csv(Dimensions_Path)
        Embeddings_Dimensions = Dimensions.iloc[0, 1]
        Annoy_Index = AnnoyIndex(Embeddings_Dimensions, 'angular')
        Annoy_Index.load(PATH_ANNOY_INDEX_TFIDF)
    elif EMBEDDING_TYPE == "DistillBERT":
        Annoy_Index = AnnoyIndex(768, 'angular')

        Annoy_Index.load(PATH_ANNOY_INDEX_DISTILBERT)
    elif EMBEDDING_TYPE == "WORD2VEC_SPACY":
        Annoy_Index = AnnoyIndex(300, 'angular')
        Annoy_Index.load(PATH_ANNOY_INDEX_SPACY)
    else:
        return jsonify({"error": "Invalid embedding type"}), 400
    return jsonify({"message": "Annoy index loaded successfully"}), 200

def Load_Title_Overview_Dataset():
    global Title_Overiview_Dataset
    
    try:
        Title_Overiview_Dataset = pd.read_csv(PATH_DATASET)

        return jsonify({"message": "Title Overview Dataset loaded successfully"}), 200

    except Exception as e:
        return jsonify({"error": "Failed to load Title Overview Dataset: {}".format(str(e))}), 500
    









TFIDF_Vectorizer_Path = "Saved_Models/Text_Based_Movie_Recommender/TFIDF/TFIDF_Vectorizer.pkl"
def Load_Model_TFIDF():
    global Model
    global TFIDF_Vectorizer_Path
    global EMBEDDING_TYPE

    if EMBEDDING_TYPE == "TFIDF":
        Model = Load_TFIDF_Vectorizer(TFIDF_Vectorizer_Path)
        return jsonify({"message": "TFIDF model loaded successfully"}), 200
    else:
        return jsonify({"error": f"Invalid embedding type. Expected 'TFIDF', but got '{EMBEDDING_TYPE}'."}), 400
    

def Load_Model_DistilBERT():
    global Model
    global EMBEDDING_TYPE

    if EMBEDDING_TYPE == "DistillBERT":
        Model = Load_DistillBERT_Model()
        print("DistillBERT model loaded successfully")
        return jsonify({"message": "DistillBERT model loaded successfully"}), 200
    else:
        return jsonify({"error": f"Invalid embedding type. Expected 'DistillBERT', but got '{EMBEDDING_TYPE}'."}), 400
    

    

def Load_Spacy_Model():
    global Model
    global EMBEDDING_TYPE

    if EMBEDDING_TYPE == "WORD2VEC_SPACY":
        Model = Load_NLP_Spacy()
        return jsonify({"message": "Spacy model loaded successfully"}), 200
    else:
        return jsonify({"error": f"Invalid embedding type. Expected 'WORD2VEC_SPACY', but got '{EMBEDDING_TYPE}'."}), 400
    





def Predict_TFIDF(Querry, Model,Title_Overiview_Dataset, Annoy_Index, Num_Neighbors=5):
    """
    Recommend movies based on a query using TF-IDF and KNN.
    
    Args:
        Querry (str): The input query text.
        Title_Overiview_Dataset (pd.DataFrame): The dataset containing movie titles and overviews.
        Annoy_Index: The Annoy index for KNN search.
        Num_Neighbors (int): The number of neighbors to consider for recommendations.
        
    Returns:
        list: A list of recommended movie titles and overviews.
    """
    

    Neighbors_Titles, Neighbors_Overviews = Recommends_Movies_TFIDF(Querry, Title_Overiview_Dataset, Annoy_Index, Model, Num_Neighbors)


    return Neighbors_Titles, Neighbors_Overviews


def Predict_DistilBERT(Querry, Model, Title_Overiview_Dataset, Annoy_Index, Num_Neighbors=5):
    """
    Recommend movies based on a query using DistillBERT and KNN.
    
    Args:
        Querry (str): The input query text.
        Title_Overiview_Dataset (pd.DataFrame): The dataset containing movie titles and overviews.
        Annoy_Index: The Annoy index for KNN search.
        Num_Neighbors (int): The number of neighbors to consider for recommendations.
        
    Returns:
        list: A list of recommended movie titles and overviews.
    """
    
    Neighbors_Titles, Neighbors_Overviews = Recommends_Movies_DistillBERT(Querry, Model, Title_Overiview_Dataset, Annoy_Index, Num_Neighbors)


    return Neighbors_Titles, Neighbors_Overviews


def Predict_Spacy(Querry, Title_Overiview_Dataset, Annoy_Index, Num_Neighbors=5):
    """
    Recommend movies based on a query using Spacy embeddings and KNN.
    
    Args:
        Querry (str): The input query text.
        Title_Overiview_Dataset (pd.DataFrame): The dataset containing movie titles and overviews.
        Annoy_Index: The Annoy index for KNN search.
        Num_Neighbors (int): The number of neighbors to consider for recommendations.
        
    Returns:
        list: A list of recommended movie titles and overviews.
    """
    

    global Model

    Neighbors_Titles, Neighbors_Overviews = Recommends_Movies_Spacy(Querry, Model,Title_Overiview_Dataset, Annoy_Index, Num_Neighbors)


    return Neighbors_Titles, Neighbors_Overviews




def Predict(Querry):
    global Model
    global Annoy_Index
    global Title_Overiview_Dataset
    global EMBEDDING_TYPE
    global Name_Label_To_Index
    global Index_To_Name_Label

    if EMBEDDING_TYPE == "TFIDF":
        Neighbors_Titles, Neighbors_Overviews = Predict_TFIDF(Querry, Model, Title_Overiview_Dataset, Annoy_Index)
    elif EMBEDDING_TYPE == "DistillBERT":
        Neighbors_Titles, Neighbors_Overviews = Predict_DistilBERT(Querry, Model, Title_Overiview_Dataset, Annoy_Index)
    elif EMBEDDING_TYPE == "WORD2VEC_SPACY":
        Neighbors_Titles, Neighbors_Overviews = Predict_Spacy(Querry, Title_Overiview_Dataset, Annoy_Index)
    else:

        return jsonify({"error": f"Invalid embedding type. Expected 'TFIDF', 'DistillBERT', or 'WORD2VEC_SPACY', but got '{EMBEDDING_TYPE}'."}), 400

    return jsonify({
        "recommended_movies_titles": Neighbors_Titles,
        "recommended_movies_descriptions": Neighbors_Overviews
    }), 200


@Api_Model_Text_Based_Movie_Recommender.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data['query']


    print("Received query:", query)
    # Check if the model is loaded
    if Model is None:
        return jsonify({"error": "Model not loaded"}), 400
    # Check if the Annoy index is loaded
    if Annoy_Index is None:
        print("Annoy index not loaded")
        return jsonify({"error": "Annoy index not loaded"}), 400
    # Check if the dataset is loaded
    if Title_Overiview_Dataset is None:
        print("Dataset not loaded")
        return jsonify({"error": "Dataset not loaded"}), 400
    # Check if the embedding type is set
    if EMBEDDING_TYPE is None:
        print("Embedding type not set")
        return jsonify({"error": "Embedding type not set"}), 400
    # Check if the query is valid
    if not isinstance(query, str) or len(query) == 0:
        print("Invalid query")
        return jsonify({"error": "Invalid query"}), 400

    print("Query:", query)

    # Call the Predict function with the query
    response = Predict(query)
    
    return response








    


    











