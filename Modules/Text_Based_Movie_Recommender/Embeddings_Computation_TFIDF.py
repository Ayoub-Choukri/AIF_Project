from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import scipy.sparse
import pickle
def Compute_Embeddings_TFIDF_List_Of_Texts(List_Of_Texts,min_df = 3, max_features = 2000,Save_Vectorizer=True, Save_Path=None):
    """
    Compute the TF-IDF embeddings for a list of texts.
    
    Args:
        List_Of_Texts (list): A list of input texts to compute embeddings for.
        
    Returns:
        np.ndarray: A 2D array of TF-IDF embeddings for the input texts.
    """
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(min_df = min_df,stop_words='english', lowercase=True, max_features=max_features)
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(List_Of_Texts)
    
    # Save the vectorizer if required
    if Save_Vectorizer and Save_Path:
        with open(Save_Path, 'wb') as f:
            pickle.dump(vectorizer, f)
            print(f"TF-IDF vectorizer saved to {Save_Path}")

        # Save the Dimensions of the TF-IDF matrix in a CSV file
        Dimensions = pd.DataFrame({'Dimension': tfidf_matrix.shape})
        Dimensions.to_csv(Save_Path.replace('.pkl', '_dimensions.csv'), index=False)
        print(f"TF-IDF matrix dimensions saved to {Save_Path.replace('.pkl', '_dimensions.csv')}")



    return tfidf_matrix



def Compute_Embeddings_Dataset_TFIDF(Dataset, Title_Column_Name, Description_Column_Name, min_df = 3, max_features = 2000,Save_Vectorizer=True,Save_Vectorizer_Path=None,Save_Embeddings=False, Save_Embeddings_Path=None):
    """
    Compute the TF-IDF embeddings for a dataset.
    
    Args:
        Dataset (pd.DataFrame): The input dataset containing text data.
        Title_Column_Name (str): The name of the title column in the dataset.
        Description_Column_Name (str): The name of the description column in the dataset.
        Save (bool): Whether to save the embeddings to a CSV file.
        Save_Path (str): The path to save the embeddings CSV file.
        
    Returns:
        np.ndarray: A 2D array of TF-IDF embeddings for the input dataset.
    """
    # Combine title and description columns
    Dataset['Combined_Text'] = Dataset[Title_Column_Name].fillna('') + " " + Dataset[Description_Column_Name].fillna('')
    
    # Compute TF-IDF embeddings
    Embeddings = Compute_Embeddings_TFIDF_List_Of_Texts(Dataset['Combined_Text'].tolist(), min_df=min_df, max_features=max_features, Save_Vectorizer=Save_Vectorizer, Save_Path=Save_Vectorizer_Path)                                       
    


    if Save_Embeddings and Save_Embeddings_Path:
        scipy.sparse.save_npz(Save_Embeddings_Path, Embeddings)  # Sauvegarde sous format sparse .npz
        print(f"TF-IDF sparse matrix saved to {Save_Embeddings_Path}")
        # Save the Dimensions of the TF-IDF matrix in a CSV file
        Dimensions = pd.DataFrame({'NB_Of_Samples': [Embeddings.shape[0]], 'NB_Of_Features': [Embeddings.shape[1]]})
        Dimensions.to_csv(Save_Embeddings_Path.replace('.npz', '_dimensions.csv'), index=False)
        print(f"TF-IDF matrix dimensions saved to {Save_Embeddings_Path.replace('.npz', '_dimensions.csv')}")


    



    return Embeddings



