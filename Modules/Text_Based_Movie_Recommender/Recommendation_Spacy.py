import pandas as pd 
Modules_Text_Based_Movie_Recommender = "Modules/Text_Based_Movie_Recommender"
import sys
sys.path.append(Modules_Text_Based_Movie_Recommender)
from KNN_Text_Based import *
import spacy
from annoy import AnnoyIndex



def Load_NLP_Spacy(model_name_spacy="en_core_web_md"):
    """
    Load the Spacy model for sentence embedding.
    
    Args:
        model_name_spacy (str): The name of the Spacy model to load.
        
    Returns:
        nlp: The loaded Spacy model.
    """
    # Load the Spacy model
    nlp = spacy.load(model_name_spacy)
    
    return nlp


def Compute_Query_Vector_Spacy(text,Model):
    """
    Compute the query vector using Spacy embeddings.
    
    Args:
        text (str): The input text to compute the query vector for.
        
    Returns:
        np.ndarray: The computed query vector.
    """
    nlp = Model 
    
    # Process the text
    doc = nlp(text)
    print(f"Doc: {doc}")
    # Get the vector representation of the text
    query_vector = doc.vector
    
    return query_vector




def Recommends_Movies_Spacy(Querry, Model,Title_Overiview_Dataset, Annoy_Index, Num_Neighbors=5):
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
    
    # Compute the query vector
    Query_Embedding = Compute_Query_Vector_Spacy(Querry,Model)
    print(f"Query_Embedding: {Query_Embedding}")
    # Get the nearest neighbors
    Neighbors_Titles, Neighbors_Overviews = Compute_KNN_Text_Based(Query_Embedding, Title_Overiview_Dataset, Annoy_Index, Num_Neighbors)

    return Neighbors_Titles, Neighbors_Overviews

if __name__ == "__main__":
    # Example usage
    # Load the dataset
    Title_Overview_Dataset_Path = "Data/Text_Based_Movie_Recommender/Title_Overview_Dataset.csv"
    Dataset = pd.read_csv(Title_Overview_Dataset_Path)
    
    # Define the column names
    Title_Column_Name = 'title'
    Description_Column_Name = 'overview'
    

    Model = Load_NLP_Spacy(model_name_spacy="en_core_web_md")


    # Load the Annoy index (assuming it's already created)
    PATH_SAVE_ANNOY_INDEX_SPACY = "Data/Text_Based_Movie_Recommender/Annoy_Indexes/Spacy/Annoy_Index.ann"
    Annoy_Index_Spacy = AnnoyIndex(300, 'angular')
    Annoy_Index_Spacy.load(PATH_SAVE_ANNOY_INDEX_SPACY)

    # Example query
    query = input("Enter a movie title or description: ")

    # Get recommendations
    recommended_titles, recommended_overviews = Recommends_Movies_Spacy(query,Model, Dataset, Annoy_Index_Spacy, Num_Neighbors=5)

    # Print the recommendations
    for i in range(len(recommended_titles)):
        print(f"=====\nRecommended Movie {i+1}:")
        print(f"Title: {recommended_titles[i]}")
        print(f"Overview: {recommended_overviews[i]}")
#     print(f"Title: {recommended_titles[i]}")


