import torch 
from tqdm.auto import tqdm
from annoy import AnnoyIndex

def Compute_Embeddings(DataLoader, Model, Device):
    """
    Compute the embeddings of the images.
    
    Parameters
    ----------
    DataLoader : torch.utils.data.DataLoader
        The DataLoader.
    Model : torch.nn.Module
        The model.
    Device : str
        The device.
        
    Returns
    -------
    Embeddings : torch.Tensor
        The embeddings.
    """
    Model.eval()
    Embeddings = []
    with torch.no_grad():
        for Images, _ in tqdm(DataLoader):
            Images = Images.to(Device)
            Embedding = Model(Images)
            Embeddings.append(Embedding)
    Embeddings = torch.cat(Embeddings)

    return Embeddings


def Create_Annoy_Index(Embeddings, Num_Trees=10, Metric='angular',Save=False, Path_Save=None):
    """
    Create an Annoy Index.
    
    Parameters
    ----------
    Embeddings : torch.Tensor
        The embeddings.
    Num_Trees : int
        The number of trees.
    Metric : str
        The metric.
        
    Returns
    -------
    Annoy_Index : AnnoyIndex
        The Annoy Index.
    """
    Num_Embeddings, Embedding_Size = Embeddings.shape
    Annoy_Index = AnnoyIndex(Embedding_Size, Metric)
    for Index in tqdm(range(Num_Embeddings), desc="Adding items to the Annoy Index"):
        Annoy_Index.add_item(Index, Embeddings[Index])
        
    Annoy_Index.build(Num_Trees)

    if Save:
        Annoy_Index.save(Path_Save)

    
    
    return Annoy_Index




def Load_Annoy_Index(Path,Embeddings_Size=2047, Metric='angular'):
    """
    Load an Annoy Index.
    
    Parameters
    ----------
    Path : str
        The path.
        
    Returns
    -------
    Annoy_Index : AnnoyIndex
        The Annoy Index.
    """
    Annoy_Index = AnnoyIndex(Embeddings_Size, Metric)
    Annoy_Index.load(Path)
    
    return Annoy_Index






