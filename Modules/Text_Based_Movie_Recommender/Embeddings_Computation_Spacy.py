import spacy
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

def Compute_Embeddings_Spacy_One_Text(NLP,Text):
    """
    Compute the embeddings for a given text using SpaCy.
    
    Args:
        text (str): The input text to compute embeddings for.
        
    Returns:
        list: A list of embeddings for each word in the text.
    """
    # Process the text with SpaCy
    # print(f"Computing embeddings for text: {Text}")
    doc = NLP(Text)
    
    # Extract the embeddings for each token
    embeddings = [token.vector for token in doc]
    
    return embeddings




def Compute_Embeddings_Spacy_List_Of_Texts(NLP, List_Of_Texts):
    """
    Compute the embeddings for a list of texts using SpaCy.
    
    Args:
        List_Of_Texts (list): A list of input texts to compute embeddings for.
        
    Returns:
        list: A list of lists, where each inner list contains the embeddings for a corresponding text.
    """
    # Process the texts with SpaCy
    
    Embeddings = []

    for Text in List_Of_Texts:
        doc = NLP(Text)
        
        # Extract the embeddings for each token
        embeddings = [token.vector for token in doc]
        
        embeddings_text = np.mean(embeddings, axis=0)
        # Append the embeddings to the list
        Embeddings.append(embeddings_text)


    return Embeddings

def Compute_Embeddings_Dataset_Spacy(NLP, Dataset, Title_Column_Name, Description_Column_Name, Save=False, Save_Path=None):

    print("Computing embeddings for the dataset...")
    Progress_Bar = tqdm(total=len(Dataset), desc="Computing embeddings", unit="text")   

    Embeddings = []

    for index, row in Dataset.iterrows():

        Title = row[Title_Column_Name]
        Description = row[Description_Column_Name]

        # Compute the embeddings for the title and description
        if pd.isna(Title) and not pd.isna(Description):
            print(f"\033[91mTitle is None for film {Description}\033[0m")
            Description_Embeddings = Compute_Embeddings_Spacy_One_Text(NLP, Description)
            Combined_Embeddings = np.mean(Description_Embeddings, axis=0)
        elif Title != None and pd.isna(Description):
            print(f"\033[91mDescription is None for film {Title}\033[0m")
            Title_Embeddings = Compute_Embeddings_Spacy_One_Text(NLP, Title)
            Combined_Embeddings = np.mean(Title_Embeddings, axis=0)
        elif Title != None and Description != None:
            Title_Embeddings = np.mean(Compute_Embeddings_Spacy_One_Text(NLP, Title),axis=0)
            Description_Embeddings = np.mean(Compute_Embeddings_Spacy_One_Text(NLP, Description),axis=0)
            Combined_Embeddings = 0.7*Title_Embeddings + 0.3*Description_Embeddings
        else:
            print(f"\033[91mBoth title and description are None for index {index}. Skipping this entry.\033[0m")
            Combined_Embeddings = np.zeros(NLP.vocab.vectors_length)  # ou None, selon ce que tu préfères

        Embeddings.append(Combined_Embeddings)
        Progress_Bar.update(1)
    Progress_Bar.close()

    # Transformer la liste de vecteurs en tableau numpy
    Embeddings_Array = np.vstack(Embeddings)

    # Ajouter chaque dimension dans une colonne
    for i in range(Embeddings_Array.shape[1]):
        Dataset[f"Embedding_{i}"] = Embeddings_Array[:, i]

    print("Embeddings computed and added as separate columns.")

    if Save and Save_Path is not None:
        Dataset.to_csv(Save_Path, index=False)
        print(f"Dataset with embeddings saved to {Save_Path}")

    return Dataset
