# models.py

from torchvision import models
import torch.nn as nn

def get_model(model_name: str, num_classes: int, pretrained: bool = False):
    """
    Loads a specified VGG model and adapts its classifier for the given number of classes.
    """
    if model_name == 'vgg11':
        model = models.vgg11_bn(pretrained=pretrained)
    elif model_name == 'vgg16':
        model = models.vgg16_bn(pretrained=pretrained)
    elif model_name == 'vgg19':
        model = models.vgg19_bn(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # Replace the final classifier layer to match the number of classes for CIFAR-100.
    # The pretrained models are for ImageNet (1000 classes).
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    return model