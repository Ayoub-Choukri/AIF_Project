from annoy import AnnoyIndex
from tqdm.auto import tqdm
import numpy as np

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
        Embedding = Embeddings[Index]

        # Sil est sparse, convert to dense
        if hasattr(Embedding, 'toarray'):
            Embedding = Embedding.toarray().flatten()

        Annoy_Index.add_item(Index, Embedding)
        
    Annoy_Index.build(Num_Trees)
    print(f"Annoy Index built with {Num_Embeddings} items and {Embedding_Size} dimensions and {Num_Trees} trees.")
    if Save:
        Annoy_Index.save(Path_Save)

    

    print(f"Nombres of elements in the Annoy Index: {Annoy_Index.get_n_items()}")
    
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
