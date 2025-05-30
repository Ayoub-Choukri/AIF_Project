

import pandas as pd 
import sys
PATH_MODULES_TEXT_BASED_MOVIE_RECOMMENDER = "Modules/Text_Based_Movie_Recommender"
sys.path.append(PATH_MODULES_TEXT_BASED_MOVIE_RECOMMENDER)

from KNN_Text_Based import Compute_KNN_Text_Based
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from annoy import AnnoyIndex







def Load_TFIDF_Vectorizer(TFIDF_Vectorizer_Path):
    """
    Load the TF-IDF vectorizer from the specified path.
    
    Args:
        TFIDF_Vectorizer_Path (str): The path to the TF-IDF vectorizer file.
        
    Returns:
        TfidfVectorizer: The loaded TF-IDF vectorizer.
    """
    with open(TFIDF_Vectorizer_Path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    return tfidf_vectorizer




def Compute_Query_Vector(text, TFIDF_Vectorizer):

    # Transform the input text to get the embedding vector
    query_vector = TFIDF_Vectorizer.transform([text])

    return query_vector



def Recommends_Movies_TFIDF(Querry, Title_Overiview_Dataset, Annoy_Index, TFIDF_Vectorizer, Num_Neighbors=5):
    """
    Recommend movies based on a query using TF-IDF and KNN.
    
    Args:
        Querry (str): The input query text.
        Title_Overiview_Dataset (pd.DataFrame): The dataset containing movie titles and overviews.
        Annoy_Index: The Annoy index for KNN search.
        TFIDF_Vectorizer_Path (str): The path to the TF-IDF vectorizer file.
        Num_Neighbors (int): The number of neighbors to consider for recommendations.
        
    Returns:
        list: A list of recommended movie titles and overviews.
    """
    
    # Compute the query vector
    Query_Embedding = Compute_Query_Vector(Querry, TFIDF_Vectorizer).toarray()[0]
    # Get the nearest neighbors


    Neighbors_Titles, Neighbors_Overviews = Compute_KNN_Text_Based(Query_Embedding, Title_Overiview_Dataset, Annoy_Index, Num_Neighbors)

    return Neighbors_Titles, Neighbors_Overviews




def Recommends_Movies_TFIDF_From_Path(Querry, Title_Overiview_Dataset, Annoy_Index, TFIDF_Vectorizer_Path, Num_Neighbors=5):
    """
    Recommend movies based on a query using TF-IDF and KNN.
    
    Args:
        Querry (str): The input query text.
        Title_Overiview_Dataset (pd.DataFrame): The dataset containing movie titles and overviews.
        Annoy_Index: The Annoy index for KNN search.
        TFIDF_Vectorizer_Path (str): The path to the TF-IDF vectorizer file.
        Num_Neighbors (int): The number of neighbors to consider for recommendations.
        
    Returns:
        list: A list of recommended movie titles and overviews.
    """
    
    # Load the TF-IDF vectorizer
    tfidf_vectorizer = Load_TFIDF_Vectorizer(TFIDF_Vectorizer_Path)

    Neighbors_Titles, Neighbors_Overviews = Recommends_Movies_TFIDF(Querry, Title_Overiview_Dataset, Annoy_Index, tfidf_vectorizer, Num_Neighbors)

    return Neighbors_Titles, Neighbors_Overviews



if __name__ == "__main__":
    # Example usage
    # Load the dataset
    Title_Overview_Dataset_Path = "Data/Text_Based_Movie_Recommender/Title_Overview_Dataset.csv"
    Dataset = pd.read_csv(Title_Overview_Dataset_Path)

    # Define the column names
    Title_Column_Name = 'title'
    Description_Column_Name = 'overview'


    # Load the Annoy index (assuming it's already created)
    Annoy_Index_Path = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Data/Text_Based_Movie_Recommender/Annoy_Indexes/Tfidf/Annoy_Index.ann"

    Dimensions_Path = "Data/Text_Based_Movie_Recommender/Embeddings/TFIDF/Embeddings_Title_Overview_Dataset_TFIDF_dimensions.csv"
    TFIDF_Vectorizer_Path = "Saved_Models/Text_Based_Movie_Recommender/TFIDF/TFIDF_Vectorizer.pkl"
    # Load the dimensions of the TF-IDF matrix
    Dimensions = pd.read_csv(Dimensions_Path)
    Embeddings_Dimensions = Dimensions.iloc[0, 1]

    Annoy_Index = AnnoyIndex(Embeddings_Dimensions, 'angular')
    Annoy_Index.load(Annoy_Index_Path)
    # Recommend movies based on a query
    Query = input("Enter a movie title or description: ")
    Neighbors_Titles,Neighbors_Overviews = Recommends_Movies_TFIDF_From_Path(Query, Dataset, Annoy_Index, TFIDF_Vectorizer_Path=TFIDF_Vectorizer_Path, Num_Neighbors=5)

    # Print the recommended movie
    for i in range(len(Neighbors_Titles)):
        print(f"====Recommendation {i+1}====")
        print(f"Title: {Neighbors_Titles[i]}")
        print(f"Overview: {Neighbors_Overviews[i]}")
        print()



    

