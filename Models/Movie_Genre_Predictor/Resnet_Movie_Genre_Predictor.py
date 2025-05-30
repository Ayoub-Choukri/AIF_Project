import torchvision.models as Models
import torch


def Get_ResNet(Pretrained=False, ResNet_Version=18, Num_Classes=1000):
    """
    Function to get a ResNet model, pretrained or not, and adapt the output layer to the number of classes.

    Parameters:
    Pretrained (bool): If True, returns a model pre-trained on ImageNet.
    ResNetVersion (int): The version of ResNet to use. Options are 18, 34, 50, 101, 152.
    Num_Classes (int): The number of output classes.

    Returns:
    Model: The ResNet model.
    """
    if ResNet_Version == 18:
        Model = Models.resnet18(pretrained=Pretrained)
    elif ResNet_Version == 34:
        Model = Models.resnet34(pretrained=Pretrained)
    elif ResNet_Version == 50:
        Model = Models.resnet50(pretrained=Pretrained)
    elif ResNet_Version == 101:
        Model = Models.resnet101(pretrained=Pretrained)
    elif ResNet_Version == 152:
        Model = Models.resnet152(pretrained=Pretrained)
    else:
        raise ValueError("Invalid ResNet version. Options are 18, 34, 50, 101, 152.")
    
    # Adapt the output layer to the number of classes
    Model.fc = torch.nn.Linear(Model.fc.in_features, Num_Classes)
    
    return Model
