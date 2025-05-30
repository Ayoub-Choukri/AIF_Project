import pandas as pd
import sys
import spacy
import numpy as np
from tqdm.auto import tqdm
import scipy.sparse
PATH_MODULES = "Modules/Text_Based_Movie_Recommender"
sys.path.append(PATH_MODULES)
from Movies_Metadata_Preprocessing import Get_Title_And_Overview_Column
from Embeddings_Computation_Spacy import Compute_Embeddings_Dataset_Spacy
from Embeddings_Computation_TFIDF import Compute_Embeddings_Dataset_TFIDF
from Creating_Annoy_Indexes import Create_Annoy_Index, Load_Annoy_Index
from Embeddings_Computation_Bert import Compute_Embeddings_DistillBert_List_Of_Texts
from Embeddings_Computation_TFIDF import Compute_Embeddings_Dataset_TFIDF
# from KNN import Compute_KNN
PATH_MODELS = "Models/Text_Based_Movie_Recommender"
sys.path.append(PATH_MODELS)
from Distill_Bert_Wrapped import DistilBERTCLSExtractor,DistilBERTSentenceEmbedder

MOVIE_METADATA_DATASET_PATH = "Data/Text_Based_Movie_Recommender/archive/movies_metadata.csv"


TITLE_OVERVIEW_DATASET_SAVING_PATH = "Data/Text_Based_Movie_Recommender/Title_Overview_Dataset.csv"

# Extract the title and overview columns from the dataset
Title_Overview_Dataset = Get_Title_And_Overview_Column(Dataset_Path=MOVIE_METADATA_DATASET_PATH, Save=True, Save_Path=TITLE_OVERVIEW_DATASET_SAVING_PATH)



COMPUTE_SPACY_EMBEDDINGS = True
TITLE_OVERVIEW_DATASET_SAVING_PATH_SPACY = "Data/Text_Based_Movie_Recommender/Embeddings/Embeddings_Title_Overview_Dataset_Spacy.csv"
if COMPUTE_SPACY_EMBEDDINGS:
    print("Computing Spacy embeddings for the dataset...")



    # Using Spacy 

    NLP = spacy.load("en_core_web_md")

    
    Compute_Embeddings_Dataset_Spacy(NLP = NLP, Dataset = Title_Overview_Dataset, Title_Column_Name='title',Description_Column_Name='overview', Save=True, Save_Path=TITLE_OVERVIEW_DATASET_SAVING_PATH_SPACY)

    print("Spacy embeddings computed and saved.")

Create_Annoy_Index_Spacy = True
Load_Embeddings_Spacy = True
PATH_SAVE_ANNOY_INDEX_SPACY = "Data/Text_Based_Movie_Recommender/Annoy_Indexes/Spacy/Annoy_Index.ann"
if Create_Annoy_Index_Spacy:
    if Load_Embeddings_Spacy:
        # Load the embeddings
        Embeddings_Spacy = pd.read_csv(TITLE_OVERVIEW_DATASET_SAVING_PATH_SPACY)
        # Embeddings_Spacy = Embeddings_Spacy.drop(columns=['Unnamed: 0']).values
    else:
        pass

    # Create the Annoy index
    # Select and Create the list of embeddings
    Embeddings_Spacy = Embeddings_Spacy.iloc[:, 2:].values
    Embeddings_Spacy = Embeddings_Spacy.astype(np.float32)
    Annoy_Index_Spacy = Create_Annoy_Index(Embeddings_Spacy, Num_Trees=10, Metric='angular', Save=True, Path_Save=PATH_SAVE_ANNOY_INDEX_SPACY)
    print("Annoy index for Spacy embeddings created and saved.")




# TF-IDF Embeddings

Compute_TFIDF_EMBEDDINGS = True




TITLE_OVERVIEW_DATASET_SAVING_PATH_TFIDF = "Data/Text_Based_Movie_Recommender/Embeddings/TFIDF/Embeddings_Title_Overview_Dataset_TFIDF.npz"
Vectorizer_Saving_Path = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Saved_Models/Text_Based_Movie_Recommender/TFIDF/TFIDF_Vectorizer.pkl"
if Compute_TFIDF_EMBEDDINGS:

    MIN_DF = 3
    MAX_FEATURES = None

    print("Computing TF-IDF embeddings for the dataset...")

    # Using TF-IDF

    
    Compute_Embeddings_Dataset_TFIDF(Dataset = Title_Overview_Dataset, Title_Column_Name='title', Description_Column_Name='overview', Save_Embeddings=True, Save_Vectorizer=True,Save_Vectorizer_Path=Vectorizer_Saving_Path, Save_Embeddings_Path=TITLE_OVERVIEW_DATASET_SAVING_PATH_TFIDF, min_df=MIN_DF, max_features=MAX_FEATURES)
    print("TF-IDF embeddings computed and saved.")



Create_Annoy_Index_TFIDF = True
Load_Embeddings_TFIDF = True
PATH_SAVE_ANNOY_INDEX_TFIDF = "Data/Text_Based_Movie_Recommender/Annoy_Indexes/Tfidf/Annoy_Index.ann"
if Create_Annoy_Index_TFIDF:
    if Load_Embeddings_TFIDF:
        # Load the embeddings
        Embeddings_TFIDF = scipy.sparse.load_npz(TITLE_OVERVIEW_DATASET_SAVING_PATH_TFIDF)
        Embeddings_TFIDF = Embeddings_TFIDF

    else:
        pass

    # Create the Annoy index
    # Select and Create the list of embeddings
    
    Annoy_Index_TFIDF = Create_Annoy_Index(Embeddings_TFIDF, Num_Trees=10, Metric='angular', Save=True, Path_Save=PATH_SAVE_ANNOY_INDEX_TFIDF)


    print("Annoy index for TF-IDF embeddings created and saved.")






# DistillBERT

Compute_DistillBERT_EMBEDDINGS = True
USE_DISTILLBERT_SENTENCE_EMBEDDER = True


CLS_TOKENS_SAVING_PATH = "Data/Text_Based_Movie_Recommender/Embeddings/DistillBERT/Embeddings_CLS_Tokens_DistillBERT.npy"
if Compute_DistillBERT_EMBEDDINGS:
    print("Computing DistillBERT embeddings for the dataset...")
    MODEL_NAME = 'distilbert-base-uncased'

    if USE_DISTILLBERT_SENTENCE_EMBEDDER:
        # Using DistillBERT Sentence Embedder
        Model = DistilBERTSentenceEmbedder(MODEL_NAME)
    else:
        # Using DistillBERT CLS Extractor
        Model = DistilBERTCLSExtractor(MODEL_NAME)



    List_Of_Texts = list(Title_Overview_Dataset['title'].fillna('EMPTY TITLE') + " " + Title_Overview_Dataset['overview'].fillna('EMPTY DESCRIPTION'))
    CLS_Embeddings = Compute_Embeddings_DistillBert_List_Of_Texts(List_Of_Texts=List_Of_Texts, Model=Model, Save=True, Save_Path=CLS_TOKENS_SAVING_PATH)
    print("DistillBERT embeddings computed and saved.")

Create_Annoy_Index_DistillBERT = True
Load_Embeddings_DistillBERT = True

PATH_SAVE_ANNOY_INDEX_DISTILLBERT = "Data/Text_Based_Movie_Recommender/Annoy_Indexes/DistillBERT/Annoy_Index.ann"
if Create_Annoy_Index_DistillBERT:
    if Load_Embeddings_DistillBERT:
        # Load the embeddings
        Embeddings_DistillBERT = np.load(CLS_TOKENS_SAVING_PATH)
        # Embeddings_DistillBERT = Embeddings_DistillBERT.drop(columns=['Unnamed: 0']).values
    else:
        pass

    # Create the Annoy index
    # Select and Create the list of embeddings
    # Embeddings_DistillBERT = Embeddings_DistillBERT.iloc[:, 2:].values
    # Embeddings_DistillBERT = Embeddings_DistillBERT.astype(np.float32)
    Annoy_Index_DistillBERT = Create_Annoy_Index(Embeddings_DistillBERT, Num_Trees=10, Metric='angular', Save=True, Path_Save=PATH_SAVE_ANNOY_INDEX_DISTILLBERT)
    print("Annoy index for DistillBERT embeddings created and saved.")




# Test 

# Annoy_Index_TFIDF_Path = PATH_SAVE_ANNOY_INDEX_TFIDF
# Annoy_Index_DistillBERT_Path = PATH_SAVE_ANNOY_INDEX_DISTILLBERT
# Annoy_Index_Spacy_Path = PATH_SAVE_ANNOY_INDEX_SPACY



# # Test TF-IDF
# Annoy_Index_Path = Annoy_Index_TFIDF_Path

# Querry_Embedding = Embeddings_TFIDF[0]