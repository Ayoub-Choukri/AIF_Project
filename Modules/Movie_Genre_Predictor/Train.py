import torch 
import torch.nn as nn
from tqdm.auto import tqdm
from Utils import *
import sys


def Train_One_Epoch(Model,Train_Loader,Optimizer,Criterion,List_Train_Losses_Per_Batches, Device):

    Progress_Bar_Batch = tqdm(Train_Loader,desc="Batches")

    Train_Loss = 0
    Num_Batches = 0
    Train_Accuracy = 0
    Train_Top3_Accuracy = 0
    for images,labels in Progress_Bar_Batch:

        images = images.to(Device)
        labels = labels.to(Device)

        Optimizer.zero_grad()

        outputs = Model(images)

        loss = Criterion(outputs,labels)

        loss.backward()

        Optimizer.step()

        Running_Loss = loss.item()

        # Calculate the accuracy
        Predictions = torch.argmax(outputs,dim=1)
        Running_Accuracy = torch.sum(Predictions == labels).item()/len(labels)

        # Calculate the top 3 accuracy
        _, Top3_Predictions = torch.topk(outputs,3,dim=1)
        Running_Top3_Accuracy = torch.sum(Top3_Predictions == labels.view(-1,1)).item()/len(labels)

        Train_Top3_Accuracy += Running_Top3_Accuracy


        Train_Accuracy += Running_Accuracy

        List_Train_Losses_Per_Batches.append(Running_Loss)

        Progress_Bar_Batch.set_description(f"Num Batches: {Num_Batches} Running Train Loss: {Running_Loss:.3f} Running Train Accuracy: {Running_Accuracy:.3f} Running Top3 Accuracy: {Running_Top3_Accuracy:.3f}")

        Train_Loss += Running_Loss

        Num_Batches += 1

    Train_Accuracy = Train_Accuracy/Num_Batches
    Train_Loss = Train_Loss/Num_Batches
    Train_Top3_Accuracy = Train_Top3_Accuracy/Num_Batches
    return Train_Loss, Train_Accuracy, Train_Top3_Accuracy, List_Train_Losses_Per_Batches


def Test_One_Epoch(Model,Test_Loader,Criterion,List_Test_Losses_Per_Batches, Device):
    Progress_Bar_Batch = tqdm(Test_Loader,desc="Batches",leave=False)

    Test_Loss = 0
    Test_Accuracy = 0
    Test_Top3_Accuracy = 0
    Num_Batches = 0

    for images,labels in Progress_Bar_Batch:

        images = images.to(Device)
        labels = labels.to(Device)

        outputs = Model(images)

        loss = Criterion(outputs,labels)

        Running_Loss = loss.item()
        List_Test_Losses_Per_Batches.append(Running_Loss)

        Test_Loss += Running_Loss

        # Calculate the accuracy
        Predictions = torch.argmax(outputs,dim=1)
        Running_Accuracy = torch.sum(Predictions == labels).item()/len(labels)

        Test_Accuracy += Running_Accuracy

        # Calculate the top 3 accuracy
        _, Top3_Predictions = torch.topk(outputs,3,dim=1)
        Running_Top3_Accuracy = torch.sum(Top3_Predictions == labels.view(-1,1)).item()/len(labels)

        if Running_Top3_Accuracy > 1:
            print("Error")
            sys.exit()
        Test_Top3_Accuracy += Running_Top3_Accuracy



        Num_Batches += 1

        Progress_Bar_Batch.set_description(f"Num Batches: {Num_Batches} Running Test Loss: {Running_Loss:.3f} Running Test Accuracy: {Running_Accuracy:.3f} Running Top3 Accuracy: {Running_Top3_Accuracy:.3f}")

    Test_Accuracy = Test_Accuracy/Num_Batches
    Test_Loss = Test_Loss/Num_Batches
    Test_Top3_Accuracy = Test_Top3_Accuracy/Num_Batches
    return Test_Loss, Test_Accuracy, Test_Top3_Accuracy, List_Test_Losses_Per_Batches


def Train(Model,Train_Loader,Test_Loader,Optimizer,Criterion,Num_Epochs, Encoding_Dict, Device):

    # Set the model to training mode
    Model.train()

    # Move the model to the device
    Model.to(Device)

    List_Train_Losses_Per_Epochs = []
    List_Test_Losses_Per_Epochs = []
    List_Train_Losses_Per_Batches = []
    List_Test_Losses_Per_Batches = []

    Progress_Bar_Epochs = tqdm(range(Num_Epochs),desc="Epochs")

    for epoch in Progress_Bar_Epochs:


        Train_Loss, Train_Accuracy, Train_Top3_Accuracy, List_Train_Losses_Per_Batches = Train_One_Epoch(Model,Train_Loader,Optimizer,Criterion,List_Train_Losses_Per_Batches,Device)

        # Plot_Images_And_Predictions(Model,Train_Loader,Nb_Images=3, Encoding_Dict=Encoding_Dict,Device=Device)

        Test_Loss, Test_Accuracy, Test_Top3_Accuracy, List_Test_Losses_Per_Batches = Test_One_Epoch(Model,Test_Loader,Criterion,List_Test_Losses_Per_Batches,Device)

        # Plot_Images_And_Predictions(Model,Test_Loader,Nb_Images=3, Encoding_Dict=Encoding_Dict,Device=Device)


        List_Train_Losses_Per_Epochs.append(Train_Loss)
        List_Test_Losses_Per_Epochs.append(Test_Loss)


        print(f"Epoch: {epoch+1}/{Num_Epochs} Train Loss: {Train_Loss:.3f} Test Loss: {Test_Loss:.3f} Train Accuracy: {Train_Accuracy:.3f} Test Accuracy: {Test_Accuracy:.3f} Train Top3 Accuracy: {Train_Top3_Accuracy:.3f} Test Top3 Accuracy: {Test_Top3_Accuracy:.3f}")

        Progress_Bar_Epochs.set_description(f"Epochs Train Loss: {Train_Loss:.3f} Test Loss: {Test_Loss:.3f} Train Accuracy: {Train_Accuracy:.3f} Test Accuracy: {Test_Accuracy:.3f} Train Top3 Accuracy: {Train_Top3_Accuracy:.3f} Test Top3 Accuracy: {Test_Top3_Accuracy:.3f}")


    return List_Train_Losses_Per_Epochs,List_Test_Losses_Per_Epochs,List_Train_Losses_Per_Batches,List_Test_Losses_Per_Batches





def Plot_Losses(List_Train_Losses_Per_Epochs,List_Test_Losses_Per_Epochs,List_Train_Losses_Per_Batches,List_Test_Losses_Per_Batches, Save = False, Save_Path = None):

    plt.figure(figsize=(15,15))

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    axes[0, 0].plot(List_Train_Losses_Per_Epochs)
    axes[0, 0].set_title("Train Losses Per Epochs")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Losses")

    axes[0, 1].plot(List_Test_Losses_Per_Epochs)
    axes[0, 1].set_title("Test Losses Per Epochs")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Losses")

    axes[1, 0].plot(List_Train_Losses_Per_Batches)
    axes[1, 0].set_title("Train Losses Per Batches")
    axes[1, 0].set_xlabel("Batches")
    axes[1, 0].set_ylabel("Losses")

    axes[1, 1].plot(List_Test_Losses_Per_Batches)
    axes[1, 1].set_title("Test Losses Per Batches")
    axes[1, 1].set_xlabel("Batches")
    axes[1, 1].set_ylabel("Losses")


    plt.show()

    if Save:
        plt.savefig(Save_Path)

    

def Save_Model(Model, Save_Path):
    torch.save(Model,Save_Path)


def Load_Model(Load_Path):
    return torch.load(Load_Path)






