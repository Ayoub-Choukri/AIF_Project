import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Compose
from torchcam.methods import GradCAM
from torchvision.transforms import ToTensor, Resize
import cv2
# === Constantes ===

SIZE = (280, 185)
# === Fonctions de chargement et de prétraitement ===

def Load_Image_PIL(Image_Path, Size=SIZE):
    Image_PIL = Image.open(Image_Path).convert("RGB")

    Image_PIL  = Image_PIL.resize(Size)  # (width, height)
    return Image_PIL

def Preprocess_Image_Grad_Cam(Image_PIL):
    Transform = Compose([
        Resize((280, 185)),
        ToTensor(),
    ])
    Image = Transform(Image_PIL)

    Image = Image.unsqueeze(0)  # Ajouter la dimension du batch

    # Redimensionner l'image si nécessaire
    # Image = Image.reshape(SIZE)
    print("Image Tensor Shape:", Image.shape)
    return Image

# === Extraction de la carte Grad-CAM ===

def Get_CAM(Model, Image_Tensor, Target_Layer, Target_Class=None):
    Cam_Extractor = GradCAM(Model, target_layer=Target_Layer)
    Output = Model(Image_Tensor)

    if Target_Class is None:
        Target_Class = Output.argmax(dim=1).item()

    Cams = Cam_Extractor(Target_Class, Output)
    Cam = list(Cams.values())[0] if isinstance(Cams, dict) else Cams[0]

    # Redimensionner
    if Cam.dim() == 2:
        Cam = Cam.unsqueeze(0)  # [1, H, W]
    Cam = F.interpolate(Cam.unsqueeze(0), size=(SIZE[0], SIZE[1]), mode='bilinear', align_corners=False)[0, 0]
    Cam = Cam.detach().cpu().numpy()
    Cam = (Cam - Cam.min()) / (Cam.max() - Cam.min() + 1e-8)  # normalisation
    return Cam

# === Visualisation avec matplotlib ===

def Visualize_GradCAM(Image, Cam, Cmap='jet', Alpha=0.5):
    plt.figure(figsize=(10, 5))
    plt.imshow(Image)
    plt.imshow(Cam, cmap=Cmap, alpha=Alpha)
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()
    plt.close()

# === Fusion de l'image et de la CAM en une seule image PIL ===

def Get_CAM_Fused_Image(Image_Tensor, Cam, Cmap='jet', Alpha=0.5):
    Image_NP = np.array(Image_Tensor)  # Convertir l'image PIL en tableau NumPy
    Image_NP = (Image_NP * 255).astype(np.uint8)
    Heatmap = plt.get_cmap(Cmap)(Cam)[:, :, :3]  # RGB heatmap
    Heatmap = (Heatmap * 255).astype(np.uint8)



    Overlay = cv2.addWeighted(Image_NP, 1 - Alpha, Heatmap, Alpha, 0)
    return Image.fromarray(Overlay)  # RGB
# === Script principal ===

if __name__ == "__main__":
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model_Path = "Saved_Models/Movie_Genre_Predictor/Movie_Genre_Predictor.pth"
    Model = torch.load(Model_Path)
    Model.to(Device).eval()

    Image_Path = "Data/MovieGenre/content/sorted_movie_posters_paligema/animation/31658.jpg"
    Image_PIL = Load_Image_PIL(Image_Path)
    Image_Tensor = Preprocess_Image_Grad_Cam(Image_PIL).to(Device)

    Target_Layer = "layer4.1.conv2"
    Cam = Get_CAM(Model, Image_Tensor, Target_Layer)


    # # Visualisation interactive
    Visualize_GradCAM(Image_Tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy() , Cam)

    # # Sauvegarde ou usage de l’image fusionnée
    Fused_Image = Get_CAM_Fused_Image(Image_Tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy(), Cam)
    # Fused_Image.save("GradCAM_Overlay.jpg")


    plt.figure(figsize=(10, 5)) 

    plt.imshow(Fused_Image)
    plt.axis('off')
    plt.title("Image avec Grad-CAM superposé")
    plt.show()
