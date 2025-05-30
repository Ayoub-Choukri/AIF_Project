


import pandas as pd


def Get_Title_And_Overview_Column(Dataset_Path,Title_Column_Name='title', Overview_Column_Name='overview',Save=False,Save_Path=None):

    Movies_Metadata_Dataset = pd.read_csv(Dataset_Path, low_memory=False)

    # Extract the 'overview' column
    Movies_Metadata_Dataset = Movies_Metadata_Dataset[[Title_Column_Name, Overview_Column_Name]]


    # Remove rows with missing values in the 'overview' column
    Movies_Metadata_Dataset = Movies_Metadata_Dataset.dropna(subset=[Title_Column_Name])
    Movies_Metadata_Dataset = Movies_Metadata_Dataset.dropna(subset=[Overview_Column_Name])



    if Save and Save_Path is not None:
        Movies_Metadata_Dataset.to_csv(Save_Path, index=False)
        print(f"Dataset saved to {Save_Path}")

        
    
    print("Title and Overview columns extracted successfully.")

    return Movies_Metadata_Dataset





    


