import matplotlib.pyplot as plt
import torch 

def Plot_Images(Images, Grid_Images_Size , figsize=(15,15),Title="Plots de quelques images du jeu de données",Encoding_Dict=None,Labels=None,Save=False,Save_Path=None):

    # Créer une figure
    fig,axes = plt.subplots(Grid_Images_Size[0],Grid_Images_Size[1],figsize=figsize)

    # Parcourir les images
    for i in range(Grid_Images_Size[0]):
        for j in range(Grid_Images_Size[1]):
            # Extraire l'image
            image = Images[i*Grid_Images_Size[1]+j]

            # Extraire l'axe
            ax = axes[i,j]

            # Afficher l'image
            ax.imshow(image.permute(1,2,0))

            # Cacher les axes
            ax.axis('off')

            # Afficher le label
            if Labels:
                ax.set_title(Labels[i*Grid_Images_Size[1]+j])

    # Afficher le titre
    plt.suptitle(Title)

    # Afficher la figure

    plt.show()

    if Save:
        plt.savefig(Save_Path)

    


def Plot_Images_During_Training(Images,Ground_Truth_Labels,Predicted_Labels,figsize = (15,15), Title = "Comparaison entre labels prédits et labels réels",Save=False,Save_Path=None):

    # Créer une figure
    fig,axes = plt.subplots(1,len(Images),figsize=figsize)

    # Parcourir les images
    # On trace l'image Ground_Truth, L'image en noir et blanc et l'image colorisée prédite
    for i in range(len(Images)):

        # Extraire l'image
        Image = Images[i].cpu()
        Ground_Truth_Label = Ground_Truth_Labels[i]
        Predicted_Label = Predicted_Labels[i]

        # Extraire l'axe
        ax = axes[i]

        # Afficher l'image
        ax.imshow(Image.permute(1,2,0))

        # Cacher les axes
        ax.axis('off')

        # Afficher le label
        ax.set_title(f"Ground Truth: {Ground_Truth_Label} ; Predicted: {Predicted_Label}")

    # Afficher le titre

    plt.suptitle(Title)

    # Afficher la figure
    plt.show()

    if Save:
        plt.savefig(Save_Path)


def Plot_Images_And_Predictions(Model,DataLoader,Nb_Images,Encoding_Dict, Device):

    Images,Labels = next(iter(DataLoader))

    Images = Images.to(Device)[:Nb_Images]
    Labels = Labels.to(Device)[:Nb_Images]

    Labels = [Encoding_Dict[Label.item()] for Label in Labels]
    
    Predictions = Model(Images)

    Predictions = torch.argmax(Predictions,dim=1)

    Predictions = [Encoding_Dict[Prediction.item()] for Prediction in Predictions]


    Plot_Images_During_Training(Images,Labels,Predictions,Title="Comparaison entre les images originales et les images colorisées prédites",figsize=(15,15))


    