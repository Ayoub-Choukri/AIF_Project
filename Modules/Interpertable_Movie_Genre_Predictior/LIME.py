import torch 
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.v2 import Compose,  Resize, ToImage, ToDtype

SIZE = (280, 185)


def Load_Image_PIL(Image_Path,size=SIZE):
    """
    Charge une image à partir d'un chemin donné et la convertit en format PIL.
    
    Args:
        Image_Path (str): Chemin de l'image à charger.
        
    Returns:
        PIL.Image: Image chargée au format PIL.
    """
    Image_PIL = Image.open(Image_Path).convert("RGB")

    # Redimensionner l'image si nécessaire
    if size:
        Image_PIL = Image_PIL.resize(size)


    return Image_PIL



def Preprocess_Image_Lime(Image_PIL):
    # Appliquer les transformations recommandées
    transform = Compose([
        Resize((280, 185)),
        ToImage(),  # Convertit PIL → image torch
        ToDtype(torch.float32, scale=True)  # Normalise en [0, 1] float32
    ])

    image_tensor = transform(Image_PIL)  # [3, H, W], float32
    print("Image Tensor Shape (Torch):", image_tensor.shape)

    # Convertir en numpy [H, W, 3], uint8 → requis pour LIME
    image_np = image_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
    image_np = (image_np * 255).astype(np.uint8)
    print("Image Shape for LIME (Numpy):", image_np.shape)

    return image_np

def predict_fn(images_np, model, device):
    model.eval()
    inputs = []

    for img in images_np:
        img_pil = Image.fromarray(img)
        tensor = v2.ToTensor()(img_pil)
        tensor = tensor.unsqueeze(0)
        inputs.append(tensor)

    batch = torch.cat(inputs).to(device)

    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    return probs.cpu().numpy()




def Get_Explanation(Image,Classifier_Function,Top_Labels=3,Hide_Color=0,Num_Samples=1000):
    """
    Crée un explainer LIME pour une image donnée.
    
    Args:
        Image (np.array): Image d'entrée.
        Classifier_Function (callable): Fonction de classification prenant un batch d'images et retournant les probabilités.
        Top_Labels (int): Nombre de labels à considérer.
        Hide_Color (int): Couleur à masquer dans l'explication.
        Num_Samples (int): Nombre d'échantillons à générer pour l'explication.
        
    Returns:
        explanation: Explication LIME de l'image.
    """
    Image_Np = np.array(Image)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=Image_Np,
        classifier_fn=Classifier_Function,
        top_labels=Top_Labels,
        hide_color=Hide_Color,
        num_samples=Num_Samples
    )
    
    return explanation


def Get_Image_And_Mask(explanation, label, positive_only=False, hide_rest=False, num_features=15, min_weight=0.05):
    """
    Récupère l'image et le masque d'une explication LIME pour un label donné.
    
    Args:
        explanation: Explication LIME de l'image.
        label (int): Label pour lequel récupérer l'image et le masque.
        positive_only (bool): Si True, ne garde que les régions positives.
        hide_rest (bool): Si True, masque les autres régions.
        num_features (int): Nombre de caractéristiques à afficher.
        min_weight (float): Poids minimum pour inclure une région dans le masque.
        
    Returns:
        temp: Image avec la superposition des régions influentes.
        mask: Masque des régions influentes.
    """
    temp, mask = explanation.get_image_and_mask(
        label=label,
        positive_only=positive_only,
        hide_rest=hide_rest,
        num_features=num_features,
        min_weight=min_weight
    )

    Image_With_Boundaries = mark_boundaries(temp / 255.0, mask)
    
    return temp, mask, Image_With_Boundaries






if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "Saved_Models/Movie_Genre_Predictor/Movie_Genre_Predictor.pth"
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    image_path = "Data/MovieGenre/content/sorted_movie_posters_paligema/action/2947.jpg"
    image_pil = Load_Image_PIL(image_path, size=SIZE)
    image_np = Preprocess_Image_Lime(image_pil)


    Explanation = Get_Explanation(
        Image=image_np,
        Classifier_Function=lambda x: predict_fn(x, model, device),
        Top_Labels=3,
        Hide_Color=0,
        Num_Samples=1000
    )

    Label = Explanation.top_labels[0]  # Prendre le label le plus probable

    temp, mask, Image_With_Boundaries = Get_Image_And_Mask(
        explanation=Explanation,
        label=Label,  # Choisir le label pour lequel on veut l'explication
        positive_only=False,
        hide_rest=False,
        num_features=15,
        min_weight=0.05
    )


    # Afficher l'image avec les frontières
    plt.figure(figsize=(10, 5))
    plt.imshow(Image_With_Boundaries)
    plt.axis('off')

    plt.title("Image avec les frontières des régions influentes")

    plt.show()

