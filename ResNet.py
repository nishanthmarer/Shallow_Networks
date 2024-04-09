"""///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Name: Nishant Marer Prabhu, Zhijun Hu, Tejal Patel
FileName: ShallowModel.py
Course: Computer Vision
Description: This file contains the model structure and returns the model created
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""
from DataLoader import DeviceCheck
import torch.nn as NN
import torch 
from torchsummary import summary

class SimpleResidualBlock(NN.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = NN.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = NN.ReLU()
        self.conv2 = NN.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = NN.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the inpu


def conv_block(in_channels, out_channels, pool=False):
    layers = [NN.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              NN.BatchNorm2d(out_channels), 
              NN.ReLU(inplace=True)]
    if pool: layers.append(NN.MaxPool2d(2))
    return NN.Sequential(*layers)

class ResNet9(NN.Module):
    def __init__(self, numChannels, classes):
        super(ResNet9,self).__init__()
        
        self.conv1 = conv_block(numChannels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = NN.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = NN.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = NN.Sequential(NN.MaxPool2d(4), 
                                        NN.Flatten(), 
                                        NN.Linear(512, classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    
"""###################################################################################"""

"""###################################################################################"""
"""The main function starts here which takes the input from the user and decides to perform training or testing of the model"""  
if __name__ =="__main__":
    #Check Device availability
    device = DeviceCheck()
    
    cnn = ResNet9(numChannels = 3,classes = 10).to(device)
    summary(cnn,(3,32,32))
    print(cnn)
    
    
    