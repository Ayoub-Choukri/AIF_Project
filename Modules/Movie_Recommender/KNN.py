from annoy import AnnoyIndex
from PIL import Image


def Compute_KNN(Query_Embedding, Data_Paths_And_Labels, Annoy_Index, Num_Neighbors=5):
    """
    Compute the k-nearest neighbors.
    
    Parameters
    ----------
    Query_Embeddings : torch.Tensor
        The query embeddings.
    Data_Paths_And_Labels : pandas.DataFrame
        The DataFrame containing the paths of the images and their corresponding labels.
    Annoy_Index : AnnoyIndex
        The Annoy Index.
    Num_Neighbors : int
        The number of neighbors.
        
    Returns
    -------
    Neighbors : pandas.DataFrame
        The DataFrame containing the paths of the images and their corresponding labels of the neighbors.
    """
    Neighbors = []

    Neighbors_Indices = Annoy_Index.get_nns_by_vector(Query_Embedding, Num_Neighbors+1)[1:]

    for Neighbor_Index in Neighbors_Indices:
        Neighbors.append(Data_Paths_And_Labels.iloc[Neighbor_Index, 0])

    

    
    return Neighbors



def Import_Images_From_Paths(Images_Paths):
    """
    Import the images.
    
    Parameters
    ----------
    Images_Paths : list
        The paths of the images.
        
    Returns
    -------
    Images : list
        The images.
    """
    Images = []

    for Image_Path in Images_Paths:
        # print(f"Image_Path: {Image_Path}")
        Image_Imported = Image.open(Image_Path)
        Images.append(Image_Imported)
    
    return Images


def Display_Images(Querry_Image, Neighbors_Images):
    """
    Display the images.
    
    Parameters
    ----------
    Querry_Image : PIL.JpegImagePlugin.JpegImageFile
        The query image.
    Neighbors_Images : list
        The neighbors images.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(Neighbors_Images), figsize=(15, 5))

    # Display the query image in the first row, centered
    axes[0, len(Neighbors_Images) // 2].imshow(Querry_Image)
    axes[0, len(Neighbors_Images) // 2].axis('off')
    axes[0, len(Neighbors_Images) // 2].set_title('Query Image')

    # Desactivate the axes other than the one of the query image
    for i in range(len(Neighbors_Images)):
        if i != len(Neighbors_Images) // 2:
            axes[0, i].axis('off')

            

    # Display the neighbors in the second row
    for i, neighbor_image in enumerate(Neighbors_Images):
        axes[1, i].imshow(neighbor_image)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Neighbor {i+1}')

    plt.tight_layout()
    plt.show()