import pandas as pd 
import sys
PATH_MODULES_TEXT_BASED_MOVIE_RECOMMENDER = "Modules/Text_Based_Movie_Recommender"
sys.path.append(PATH_MODULES_TEXT_BASED_MOVIE_RECOMMENDER)
from KNN_Text_Based import *
import torch 
from annoy import AnnoyIndex
import sys
PATH_MODELS = "Models/Text_Based_Movie_Recommender"
sys.path.append(PATH_MODELS)
from Distill_Bert_Wrapped import DistilBERTCLSExtractor ,DistilBERTSentenceEmbedder




def Load_DistillBERT_Model():
    """
    Load the DistillBERT model for sentence embedding.
    
    Returns:
        DistilBERTSentenceEmbedder: The loaded DistillBERT model.
    """
    # Load the DistillBERT model
    Model = DistilBERTSentenceEmbedder()
    return Model




def Compute_Query_Vector_Spacy(Text,Model):
    Model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model.to(device)

    # Process the text
    with torch.no_grad():
        embedding = Model([Text])  # single text → wrapped as list
        Query_Embedding = embedding.squeeze(0).cpu().numpy()  # remove batch dim
    return Query_Embedding



def Recommends_Movies_DistillBERT(Querry,Model, Title_Overiview_Dataset, Annoy_Index, Num_Neighbors=5):

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
    
    # Compute the query vector
    Query_Embedding = Compute_Query_Vector_Spacy(Querry, Model)
    # Get the nearest neighbors
    Neighbors_Titles, Neighbors_Overviews = Compute_KNN_Text_Based(Query_Embedding, Title_Overiview_Dataset, Annoy_Index, Num_Neighbors)

    return Neighbors_Titles, Neighbors_Overviews



if __name__ == "__main__":
    # Example usage
    # Load the dataset
    Title_Overview_Dataset_Path = "Data/Text_Based_Movie_Recommender/Title_Overview_Dataset.csv"
    Dataset = pd.read_csv(Title_Overview_Dataset_Path)

    # Load the Model 
    
    Model = DistilBERTSentenceEmbedder()    

    # Load the Annoy index (assuming it's already created)
    PATH_SAVE_ANNOY_INDEX = "Data/Text_Based_Movie_Recommender/Annoy_Indexes/DistillBERT/Annoy_Index.ann"

    Annoy_Index = AnnoyIndex(768, 'angular')

    Annoy_Index.load(PATH_SAVE_ANNOY_INDEX)
    # Example query
    query = input("Enter a movie title or description: ")

    # Get recommendations
    recommended_titles, recommended_overviews = Recommends_Movies_DistillBERT(query, Model, Dataset, Annoy_Index, Num_Neighbors=5)

    # Print the recommended movie titles and overviews
    for i in range(len(recommended_titles)):
        print(f"=====\nRecommended Movie {i+1}:")
        print(f"Title: {recommended_titles[i]}")
        print(f"Overview: {recommended_overviews[i]}")
        print("-" * 50)
