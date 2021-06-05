from torch import nn
from torchvision import models


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    model_type = config.model.model_type
    num_classes = config.dataset.num_of_classes
    if model_type == 'resnet34':
        print("ResNet34")
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception('model type is not supported:', model_type)
    model.to('cuda')
    return model
