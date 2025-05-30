import torch
import numpy as np
from tqdm.auto import tqdm

def Compute_Embeddings_DistillBert_List_Of_Texts(List_Of_Texts, Model,Save=False, Save_Path=None):
    """
    Compute DistilBERT CLS embeddings for a list of texts, one by one, with a progress bar.
    
    Args:
        List_Of_Texts (list): A list of input texts.
        Model (DistilBERTCLSExtractor): The wrapped DistilBERT model.
        Save (bool): Whether to save the result to disk.
        Save_Path (str): Path to save the .npy file if Save is True.
        
    Returns:
        np.ndarray: A 2D array (batch_size, hidden_dim) of CLS embeddings.
    """
    Model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model.to(device)
    
    all_embeddings = []

    for text in tqdm(List_Of_Texts, desc="Computing CLS embeddings"):
        with torch.no_grad():
            embedding = Model([text])  # single text → wrapped as list
            all_embeddings.append(embedding.squeeze(0).cpu().numpy())  # remove batch dim

    embeddings_np = np.vstack(all_embeddings)  # (N, hidden_dim)

    if Save and Save_Path:
        np.save(Save_Path, embeddings_np)
        print(f"Embeddings saved to {Save_Path}")
    
    return embeddings_np
