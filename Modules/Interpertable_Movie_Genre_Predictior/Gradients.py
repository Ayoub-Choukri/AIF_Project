import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import ToTensor
from torchvision.transforms.v2 import ToImage, ToDtype, Compose, Resize
import cv2

SIZE = (280, 185)  # (width, height) pour PIL



def Load_Image_PIL(Image_Path, size=SIZE):
    image_pil = Image.open(Image_Path).convert("RGB")
    if size:
        image_pil = image_pil.resize(size)
    return image_pil


def Preprocess_Image_Gradients(image_pil):
    image_tensor = ToTensor()(image_pil).unsqueeze(0)  # Shape: [1, 3, H, W]
    image_tensor.requires_grad_(True)

    transform = Compose([
        ToImage(),
        Resize((280, 185)),
        ToDtype(torch.float32, scale=True),
    ])

    image_tensor = transform(image_tensor)


    return image_tensor

def Get_Gradients(model, image_tensor, device):
    model.eval()

    # S'assurer que l'image est un leaf tensor
    image_tensor = image_tensor.clone().detach().requires_grad_(True).to(device)

    if image_tensor.dim() == 3:  # [C, H, W]
        image_tensor = image_tensor.unsqueeze(0)  # -> [1, C, H, W]
    elif image_tensor.dim() == 4 and image_tensor.size(0) == 1:
        pass  # Already [1, C, H, W]
    else:
        raise ValueError(f"Unexpected image_tensor shape: {image_tensor.shape}")

    output = model(image_tensor)  # [1, num_classes]
    predicted_label = output.argmax(dim=1).item()

    # ⛔ Ne pas appeler backward() ici
    gradients = torch.autograd.grad(
        outputs=output[0, predicted_label],
        inputs=image_tensor,
        retain_graph=False  # tu n'as pas besoin du graphe après
    )[0][0]  # [C, H, W]

    return gradients.cpu(), predicted_label




def Get_Image_Fused_With_Gradients(image_tensor, gradients, alpha=0.5):
    """
    Superpose les gradients (heatmap) sur l'image originale.
    """
    # Convertir [C, H, W] → [H, W, C]
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)


    # Traitement des gradients : [C, H, W] → [H, W]
    gradients = gradients.detach().cpu().numpy()
    gradients = np.abs(gradients)
    gradients = np.mean(gradients, axis=0)  # [H, W]

    # Normalisation
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)

    # Création du heatmap RGB
    heatmap = plt.cm.jet(gradients)[:, :, :3]  # [H, W, 3], valeurs entre 0 et 1
    heatmap = (heatmap * 255).astype(np.uint8)  # [0,255]


    # Redimensionner si nécessaire (sécurité)
    if heatmap.shape[:2] != image_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

    # Fusion des images
    Overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)

    return Image.fromarray(Overlay)



def Visualize_Gradients(image_pil, gradients, cmap='jet', alpha=0.5):
    """
    Affiche l’image originale avec superposition du heatmap des gradients.
    """
    image_np = np.array(image_pil.resize(SIZE))

    # Réduction des gradients [C, H, W] -> [H, W]
    gradients = gradients.detach().cpu().numpy()
    gradients = np.abs(gradients)
    gradients = np.mean(gradients, axis=0)  # [H, W]

    # Normalisation pour affichage
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)

    plt.figure(figsize=(10, 5))
    plt.imshow(image_np)
    plt.imshow(gradients, cmap=cmap, alpha=alpha)
    plt.axis('off')
    plt.title("Carte de chaleur des gradients")
    plt.show()
    plt.close()



if __name__ == "__main__":
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger le modèle
    model_path = "Saved_Models/Movie_Genre_Predictor/Movie_Genre_Predictor.pth"
    model = torch.load(model_path)
    model.to(device)

    # Charger l’image
    image_path = "Data/MovieGenre/content/sorted_movie_posters_paligema/action/135.jpg"
    image_pil = Load_Image_PIL(image_path)

    # Prétraitement
    image_tensor = Preprocess_Image_Gradients(image_pil).to(device)

    gradients, predicted_label = Get_Gradients(model, image_tensor, device)

    # Visualisation
    # Visualize_Gradients(image_pil, gradients)


    Image_Fused = Get_Image_Fused_With_Gradients(image_tensor.cpu().detach().squeeze(0), gradients, alpha=0.5)


    # plt.figure(figsize=(10, 5))

    # plt.imshow(Image_Fused)

    # plt.show()


