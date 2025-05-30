import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
# Import sampler 
from torch.utils.data import WeightedRandomSampler

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, Data_Paths,Labels,Data_Paths_And_Labels, Transform=None):
        self.Data_Paths = Data_Paths
        self.Labels = Labels
        self.Data_Paths_And_Labels = Data_Paths_And_Labels
        self.Transform = Transform

    def __len__(self):
        return len(self.Data_Paths_And_Labels)

    def __getitem__(self, Index):
        Img_Path, Label = self.Data_Paths_And_Labels[Index]
        Image_File = Image.open(Img_Path).convert("RGB")

        if self.Transform:
            Image_File = self.Transform(Image_File)

        return Image_File, Label

Name_Label_To_Index = {"action" : 0 , "animation" : 1, "comedy" : 2, "documentary" : 3, "drama" : 4 , "fantasy" : 5 , "horror" : 6  , "romance" : 7
                       ,"science Fiction" : 8 , "thriller" : 9}

Index_To_Name_Label = {v:k for k,v in Name_Label_To_Index.items() }

def Prepare_Dataset(Data_Path, Transform,Name_Label_To_Index=Name_Label_To_Index, Test_Size=0.2, Random_State=42):
    Data_Paths_And_Labels = []
    Data_Paths = []
    Labels = []
    # Créer un dictionnaire Label -> Index

    # Collecter toutes les images avec leur Label
    for Folder, Label in Name_Label_To_Index.items():
        Folder_Path = os.path.join(Data_Path, Folder)
        if os.path.isdir(Folder_Path):
            for File in os.listdir(Folder_Path):
                if File.endswith(('.jpg', '.png', '.jpeg')):
                    Data_Paths.append(os.path.join(Folder_Path, File))
                    Labels.append(Label)
                    Data_Paths_And_Labels.append((os.path.join(Folder_Path, File), Label))

    # Extraction des chemins et Labels
    Paths, Labels = zip(*Data_Paths_And_Labels)

    # Séparation stratifiée des données
    Train_Paths, Test_Paths, Train_Labels, Test_Labels = train_test_split(
        Paths, Labels, test_size=Test_Size, stratify=Labels, random_state=Random_State
    )

    # Création des Datasets
    Train_Dataset = Custom_Dataset(Train_Paths, Train_Labels,list(zip(Train_Paths, Train_Labels)), Transform=Transform)
    Test_Dataset = Custom_Dataset(Test_Paths, Test_Labels,list(zip(Test_Paths, Test_Labels)), Transform=Transform)


    
    return Train_Paths,Train_Labels,Data_Paths_And_Labels,Train_Dataset, Test_Dataset



def Create_DataLoaders(Labels,Train_Dataset, Test_Dataset, Balance_Classes=False,Batch_Size=32, Num_Workers=2, Shuffle=True):

    if not Balance_Classes:
        Labels_Counts = np.bincount(Labels)
        Weights = 1.0 / (Labels_Counts * len(Labels_Counts))

        Sample_Weights = [Weights[Label] for Label in Labels]

        Sampler = WeightedRandomSampler(weights=Sample_Weights, num_samples=len(Sample_Weights), replacement=True)

        Train_Loader = DataLoader(Train_Dataset, batch_size=Batch_Size, num_workers=Num_Workers, sampler=Sampler)
    else:
        Train_Loader = DataLoader(Train_Dataset, batch_size=Batch_Size, shuffle=Shuffle, num_workers=Num_Workers)
    Test_Loader = DataLoader(Test_Dataset, batch_size=Batch_Size, shuffle=Shuffle, num_workers=Num_Workers)
    return Train_Loader, Test_Loader





if __name__ == "__main__":
    PATH_DATA ="/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Data/MovieGenre/content/sorted_movie_posters_paligema"

    TRANSFORM = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    Data_Paths_And_Labels, Train_Dataset, Test_Dataset = Prepare_Dataset(Data_Path=PATH_DATA, Transform=TRANSFORM, Test_Size=0.2, Random_State=42)


    print("Nombre d'images dans le dataset: ", len(Data_Paths_And_Labels))
    print("Nombre d'images dans le Train Dataset: ", len(Train_Dataset))

    print("Nombre d'images dans le Test Dataset: ", len(Test_Dataset))


