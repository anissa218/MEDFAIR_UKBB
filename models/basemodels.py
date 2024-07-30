import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
import torch


class cusResNet18(nn.Module):    
    def __init__(self, n_classes, pretrained = True):
        super(cusResNet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        
        resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)
        self.avgpool = resnet.avgpool
        
        self.returnkey_avg = 'avgpool'
        self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'avgpool': self.returnkey_avg, 'fc': self.returnkey_fc})


    def forward(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_avg].squeeze()

    def inference(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_avg].squeeze()
    
    
class cusResNet50(cusResNet18):    
    def __init__(self, n_classes, pretrained = True):
        super(cusResNet50, self).__init__(n_classes, pretrained)
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)

        self.avgpool = resnet.avgpool
        self.returnkey_avg = 'avgpool'
        self.returnkey_fc = 'fc'
        self.body = create_feature_extractor(
            resnet, return_nodes={'avgpool': self.returnkey_avg, 'fc': self.returnkey_fc})
    

class cusDenseNet121(cusResNet18):
    def __init__(self, n_classes, pretrained = True, disentangle = False):
        super(cusDenseNet121, self).__init__(n_classes, pretrained)
        resnet = torchvision.models.densenet121(pretrained=pretrained)
        
        resnet.classifier = nn.Linear(resnet.classifier.in_features, n_classes)
       
        self.returnkey_fc = 'classifier'
        self.body = create_feature_extractor(
            resnet, return_nodes={'classifier': self.returnkey_fc})
    
    def forward(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_fc]

    def inference(self, x):
        outputs = self.body(x)
        return outputs[self.returnkey_fc], outputs[self.returnkey_fc]


class MLPclassifer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPclassifer, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, output_dim)
        
    def forward(self,x):
        x = self.relu(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        return x    

class SimpleCNN(nn.Module):
    def __init__(self,n_classes,pretrained = False):
        super(SimpleCNN, self).__init__() # changed in_ncahnnels
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7,n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        features = x.view(x.size(0), -1)
        output = self.out(features)
        return output, features
    
    def inference(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        features = x.view(x.size(0), -1)
        output = self.out(features)
        return output, features


class InceptionV3(cusResNet18):
    def __init__(self, n_classes, pretrained = True, disentangle = False):
        super(InceptionV3, self).__init__(n_classes, pretrained)
        net = torchvision.models.inception_v3(pretrained=pretrained)
        
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        
        self.returnkey_avg = 'avgpool'
        self.returnkey_fc = 'classifier'
        self.body = net

        print('using InceptionV3')
    
    def forward(self, x):
        outputs = self.body(x)
        if isinstance(outputs, tuple): # <-- inception output is a tuple (x, aux) during training but not in model.eval()
            outputs = outputs[0]
        return outputs, outputs

    def inference(self, x):
        outputs = self.body(x)
        self.feature_extractor = create_feature_extractor(self.body, return_nodes={'avgpool': self.returnkey_avg, 'fc': self.returnkey_fc})
        if isinstance(outputs, tuple): # <-- inception output is a tuple (x, aux) during training but not in model.eval()
            outputs = outputs[0]
        features = self.feature_extractor(x) # to get feature extraction to analyse image reprsentations
        return outputs, features[self.returnkey_avg].squeeze()