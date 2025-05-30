import os
import torch
import torch.nn as nn
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from annoy import AnnoyIndex


PATH_MODULES = "Modules/Movie_Recommender"
sys.path.append(PATH_MODULES)

PATH_MODELS = "Models/Movie_Recommender"
sys.path.append(PATH_MODELS)


from Resnet import Get_ResNet_Movie_Recommender

from Preprocessing import Create_Paths_DataFrame, Custom_Image_Dataset, Create_One_DataLoader
from Embeddings_Computing import Compute_Embeddings, Create_Annoy_Index , Load_Annoy_Index
from KNN import Compute_KNN, Import_Images_From_Paths, Display_Images



Name_Label_To_Index = {"action" : 0 , "animation" : 1, "comedy" : 2, "documentary" : 3, "drama" : 4 , "fantasy" : 5 , "horror" : 6  , "romance" : 7
                          ,"science Fiction" : 8 , "thriller" : 9}


Index_To_Name_Label = {v:k for k,v in Name_Label_To_Index.items() }




# Initialize the device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Data_Path = 'Data/MovieGenre/content/sorted_movie_posters_paligema'
SIZE = (280, 185)
Transform = transforms.Compose([transforms.Resize((280, 185)), transforms.ToTensor()])


# Create the Model 
model = Get_ResNet_Movie_Recommender(Pretrained=True, ResNet_Version=50, Num_Classes=len(Name_Label_To_Index))
model.to(DEVICE)





COMPUTE_EMBEDDINGS = False

if COMPUTE_EMBEDDINGS:

    # Create Data_Paths_And_Labels
    Data_Paths_And_Labels = Create_Paths_DataFrame(Data_Path, Name_Label_To_Index)

    # Save the Data_Paths_And_Labels
    SAVING_PATH = "Data/Movie_Recommender/Data_Paths_And_Labels/Data_Paths_And_Labels.csv"
    Data_Paths_And_Labels.to_csv(SAVING_PATH, index=False)

    # Create the DataLoader
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    SHUFFLE = False
    Dataset_To_Load = Custom_Image_Dataset(Data_Paths_And_Labels, Transform)
    Data_Loader = Create_One_DataLoader(Dataset_To_Load, Batch_Size=BATCH_SIZE, Num_Workers=NUM_WORKERS, Shuffle=SHUFFLE)


    # Compute the embeddings
    Embeddings = Compute_Embeddings(Data_Loader, model, DEVICE)

    # Add the embeddings to the Data_Paths_And_Labels
    Embeddings = Embeddings.cpu().numpy()
    Embeddings = pd.DataFrame(Embeddings)
    Data_Paths_And_Labels = pd.concat([Data_Paths_And_Labels, Embeddings], axis=1)

    Embeddings = np.array(Embeddings)

    # Save the Data_Paths_And_Labels
    Data_Paths_And_Labels.to_csv(SAVING_PATH, index=False)

else:
    Data_Paths_And_Labels = pd.read_csv("Data/Movie_Recommender/Data_Paths_And_Labels/Data_Paths_And_Labels.csv")

    Embeddings = Data_Paths_And_Labels.iloc[:, 3:].values

    




COMPUTE_ANNOY_INDEX = False
PATH_SAVE_ANNOY_INDEX = "Data/Movie_Recommender/Annoy_Index/Annoy_Index.ann"
NUM_TREES = 10
METRIC = 'angular'
LOAD_ANNOY_INDEX = True
# Compute the Annoy Index

if COMPUTE_ANNOY_INDEX : 
    Annoy_Index = Create_Annoy_Index(Embeddings, Num_Trees=NUM_TREES, Metric=METRIC, Save=True, Path_Save=PATH_SAVE_ANNOY_INDEX)
elif LOAD_ANNOY_INDEX:
    Annoy_Index = AnnoyIndex(Embeddings.shape[1], METRIC)
    Annoy_Index.load(PATH_SAVE_ANNOY_INDEX)
else:
    pass



# Test
Index_Query = 4
Query_Embeddings = Embeddings[Index_Query]

Num_Neighbors = 5

Neighbors_Paths = Compute_KNN(Query_Embeddings, Data_Paths_And_Labels, Annoy_Index, Num_Neighbors=Num_Neighbors)


Query_Image = Import_Images_From_Paths([Data_Paths_And_Labels.iloc[Index_Query, 0]])[0]

Neighbors_Images = Import_Images_From_Paths(Neighbors_Paths)



Display_Images(Query_Image, Neighbors_Images)


# Display the images
















