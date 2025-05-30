import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import pandas as pd


Name_Label_To_Index = {"action" : 0 , "animation" : 1, "comedy" : 2, "documentary" : 3, "drama" : 4 , "fantasy" : 5 , "horror" : 6  , "romance" : 7
                       ,"science Fiction" : 8 , "thriller" : 9}

Index_To_Name_Label = {v:k for k,v in Name_Label_To_Index.items() }


def Create_Paths_DataFrame(Data_Path, Name_Label_To_Index=Name_Label_To_Index):
    """
    Create a DataFrame containing the paths of the images and their corresponding labels.
    
    Parameters
    ----------
    Data_Path : str
        The path to the folder containing the images.
        
    Returns
    -------
    Data_Paths_And_Labels : pandas.DataFrame
        The DataFrame containing the paths of the images and their corresponding labels.
    """
    
    Data_Paths = []
    Labels = []
    Labels_Encoded = []
    
    for Folder, Label in Name_Label_To_Index.items():
        Folder_Path = os.path.join(Data_Path, Folder)
        if os.path.isdir(Folder_Path):
            for File in os.listdir(Folder_Path):
                if File.endswith(('.jpg', '.png', '.jpeg')):
                    Data_Paths.append(os.path.join(Folder_Path, File))
                    Labels.append(Folder)
                    Labels_Encoded.append(Label)

    Data_Paths_And_Labels = pd.DataFrame({"Path" : Data_Paths, "Label" : Labels, "Label_Encoded" : Labels_Encoded})

    return Data_Paths_And_Labels








class Custom_Image_Dataset(Dataset):
    def __init__(self, Data_Paths_And_Labels, Transform=None):
        """
        Custom Dataset class for images.
        
        Parameters
        ----------
        Data_Paths_And_Labels : pandas.DataFrame
            The DataFrame containing the paths of the images and their corresponding labels.
        Transform : torchvision.transforms
            The transformations to apply to the images.
        """
        self.Data_Paths_And_Labels = Data_Paths_And_Labels
        self.Transform = Transform
        
    def __len__(self):
        return len(self.Data_Paths_And_Labels)
    
    def __getitem__(self, Index):
        Image_Path = self.Data_Paths_And_Labels.iloc[Index, 0]
        Image_Imported = Image.open(Image_Path)
        Label = self.Data_Paths_And_Labels.iloc[Index, 2]
        
        if self.Transform:
            Image_Imported = self.Transform(Image_Imported)
        
        return Image_Imported, Label
    


def Create_One_DataLoader(Dataset, Batch_Size=32, Num_Workers=2, Shuffle=False):
    """
    Create a DataLoader for a given Dataset.
    
    Parameters
    ----------
    Labels : list
        The list of labels.
    Dataset : torch.utils.data.Dataset
        The Dataset.
    Batch_Size : int
        The batch size.
    Num_Workers : int
        The number of workers.
    Shuffle : bool
        If True, shuffle the data.
        
    Returns
    -------
    DataLoader : torch.utils.data.DataLoader
        The DataLoader.
    """
    DataLoader_To_Return = DataLoader(Dataset, batch_size=Batch_Size, num_workers=Num_Workers, shuffle=Shuffle)
    
    return DataLoader_To_Return






if __name__ == "__main__":
    Data_Path = "/home/ayoubchoukri/Etudes/5A/S2/AI_Frameworks/Projet/Data/MovieGenre/content/sorted_movie_posters_paligema"
    Data_Paths_And_Labels = Create_Paths_DataFrame(Data_Path)
    print(Data_Paths_And_Labels.head())
    print(Data_Paths_And_Labels.shape)


    

