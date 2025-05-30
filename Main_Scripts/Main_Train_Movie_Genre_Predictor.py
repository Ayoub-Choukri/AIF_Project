import torch
import torch.nn as nn
from torchvision import transforms
import sys


PATH_MODULES = "Modules/Movie_Genre_Predictor"
sys.path.append(PATH_MODULES)

PATH_MODELS = "Models/Movie_Genre_Predictor"

sys.path.append(PATH_MODELS)


from Preprocessing import *
from Utils import *
from Train import *
from Resnet_Movie_Genre_Predictor import *


# HYPERPARAMETERS
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SHUFFLE = True
TEST_SIZE = 0.05
DATA_PATH = 'Data/MovieGenre/content/sorted_movie_posters_paligema'
SIZE = (280, 185)
WEIGHT_DECAY = 2e-3
TRANSFORM = transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor()])
RESNET_VERSION = 18

Name_Label_To_Index = {"action" : 0 , "animation" : 1, "comedy" : 2, "documentary" : 3, "drama" : 4 , "fantasy" : 5 , "horror" : 6  , "romance" : 7
                       ,"science Fiction" : 8 , "thriller" : 9}

Index_To_Name_Label = {v:k for k,v in Name_Label_To_Index.items() }

print(" TRAINING THE MODEL ON DEVICE: ", DEVICE)

# Create the datasets
Data_Paths,Labels,Data_Paths_And_Labels,Train_Dataset, Test_Dataset = Prepare_Dataset(Data_Path=DATA_PATH, Transform=TRANSFORM, Test_Size=TEST_SIZE, Random_State=42)

# Create the DataLoaders
Train_Loader, Test_Loader = Create_DataLoaders(Labels=Labels,Train_Dataset = Train_Dataset, Test_Dataset = Test_Dataset, Batch_Size = BATCH_SIZE, Num_Workers = NUM_WORKERS, Shuffle = SHUFFLE)

# Create the model
model = Get_ResNet(Pretrained=True, ResNet_Version=RESNET_VERSION, Num_Classes=len(Name_Label_To_Index))

# Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Create the criterion
criterion = nn.CrossEntropyLoss()

# Train the model
List_Train_Losses_Per_Epochs,List_Test_Losses_Per_Epochs,List_Train_Losses_Per_Batches,List_Test_Losses_Per_Batches = Train(Model=model, Train_Loader=Train_Loader, Test_Loader=Test_Loader, Optimizer=optimizer, Criterion=criterion, Encoding_Dict=Index_To_Name_Label, Num_Epochs=NUM_EPOCHS, Device=DEVICE)


# Save the model
SAVE_MODEL_PATH = "Saved_Models/Movie_Genre_Predictor/Movie_Genre_Predictor.pth"
SAVE_MODEL_WEIGHTS_PATH = "Saved_Models/Movie_Genre_Predictor/Movie_Genre_Predictor_WEIGHTS.pth"
torch.save(model, SAVE_MODEL_PATH)
torch.save(model.state_dict(), SAVE_MODEL_WEIGHTS_PATH)



# Plot the losses
SAVE_LOSSES_PATH = "Saved_Metrics/Losses/Losses.png"


Plot_Losses(List_Train_Losses_Per_Epochs,List_Test_Losses_Per_Epochs,List_Train_Losses_Per_Batches,List_Test_Losses_Per_Batches, Save = True, Save_Path = SAVE_LOSSES_PATH)


