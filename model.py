import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


def arch_select(model_structure,pretrained,n_classes):

    if model_structure == 'densenet161':
        model_fl = Densenet161_fl(pretrained,n_classes)
    if model_structure == 'inceptionv3':   
        model_fl = InceptionV3_fl(pretrained,n_classes)
    if model_structure == 'resnet101':   
        model_fl = Resnet101_fl(pretrained,n_classes)
    if model_structure == 'resnext101':   
        model_fl = Resnext101_32x8d_fl(pretrained,n_classes)
    if model_structure == 'mobilenetv2':   
        model_fl = MobilenetV2_fl(pretrained,n_classes)
    if model_structure == 'vgg16bn':   
        model_fl = Vgg16_bn_fl(pretrained,n_classes)
    if model_structure == 'efficient-b4':   
        model_fl = Efficient_B4(pretrained,n_classes)
    if model_structure == 'efficient-b5':   
        model_fl = Efficient_B5(pretrained,n_classes)
    if model_structure == 'resnet50':   
        model_fl = Resnet50_fl(pretrained,n_classes)

    return model_fl


def Efficient_B4(pretrained, n_classes):
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Identity()
    net_fl = nn.Sequential(
            nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, n_classes)
            )
    model._fc = net_fl
    
    return model


def Efficient_B5(pretrained, n_classes):
    model = EfficientNet.from_pretrained('efficientnet-b5')
    model._fc = nn.Identity()
    net_fl = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, n_classes)
            )
    model._fc = net_fl
    
    return model


def Resnet50_fl(pretrained, n_classes):
    resnet101 = models.resnet50(pretrained)
    resnet101.fc = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, n_classes)
        )
    resnet101.fc = net_fl
    
    return resnet101


def InceptionV3_fl(pretrained, n_classes):
    inception_v3 = models.inception_v3(pretrained)
    inception_v3.fc = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, n_classes)
        )
    inception_v3.fc = net_fl
    
    return inception_v3


def Densenet161_fl(pretrained, n_classes):
    densenet161 = models.densenet161(pretrained)
    densenet161.classifier = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(2208, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, n_classes)
        )
    densenet161.classifier = net_fl
    
    return densenet161


def Resnet101_fl(pretrained, n_classes):
    resnet101 = models.resnet101(pretrained)
    resnet101.fc = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, n_classes)
        )
    resnet101.fc = net_fl
    
    return resnet101


def Resnext101_32x8d_fl(pretrained, n_classes):
    resnext101_32x8d = models.resnext101_32x8d(pretrained)
    resnext101_32x8d.fc = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, n_classes)
        )
    resnext101_32x8d.fc = net_fl
    
    return resnext101_32x8d


def MobilenetV2_fl(pretrained, n_classes):
    mobilenet_v2 = models.mobilenet_v2(pretrained)
    mobilenet_v2.classifier = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, n_classes)
        )
    mobilenet_v2.classifier = net_fl
    
    return mobilenet_v2



def Vgg16_bn_fl(pretrained, n_classes):
    vgg16_bn = models.vgg16_bn(pretrained)
    vgg16_bn.classifier = nn.Identity()
    net_fl = nn.Sequential(
        nn.Linear(25088, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64), 
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, n_classes)
        )
    vgg16_bn.classifier = net_fl
    
    return vgg16_bn

