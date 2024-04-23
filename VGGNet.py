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

Conv1LayerOutSizeCIFAR = 32
#CNN Layer Settings
Conv1Filter_Size = 5
Conv2Filter_Size = 3
Conv1Stride = 1
Conv2Stride = 1
Conv1OutputChannel = 32
Conv2OutputChannel = 64
Conv3OutputChannel = 128
Conv4OutputChannel = 256
ValidationSize = 0.25
dropout = 0.1
x = torch.zeros((8,8,64))

class CIFARNet(NN.Module):
    def __init__(self, numChannels, classes):
        super(CIFARNet,self).__init__()
        #VGG Type Network
        #First Layer as per requirement is 
        #F = 5x5, S = 1, K = 32, Input size = 32x32, Output Size = 32x32x32
        self.Conv1 = NN.Conv2d(in_channels = numChannels, 
                                out_channels = Conv1OutputChannel, 
                                kernel_size = Conv1Filter_Size, 
                                stride = Conv1Stride,
                                padding = 2)
        self.BatchNorm1 = NN.BatchNorm2d(Conv1OutputChannel)
        self.reluConv1 = NN.ReLU()
        
        self.Conv2 = NN.Conv2d(in_channels = Conv1OutputChannel, 
                                out_channels = Conv1OutputChannel, 
                                kernel_size = Conv1Filter_Size, 
                                stride = Conv2Stride,
                                padding = 2)
        self.BatchNorm2 = NN.BatchNorm2d(Conv1OutputChannel)
        self.reluConv2 = NN.ReLU()
        
        self.MaxPoolConv1 = NN.MaxPool2d((2,2), stride = 2)
        self.DropOut1 = NN.Dropout2d(p=dropout)
        ##########################################################################################
        #Output size at this point is 16x16x32
        #Here the filter sizes is changed from 5 to 3 for better extraction of the features in the image
        self.Conv3 = NN.Conv2d(in_channels = Conv1OutputChannel, 
                                out_channels = Conv2OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv1Stride,
                                padding = 1)
        self.BatchNorm3 = NN.BatchNorm2d(Conv2OutputChannel)
        self.reluConv3 = NN.ReLU()
        
        self.Conv4 = NN.Conv2d(in_channels = Conv2OutputChannel, 
                                out_channels = Conv2OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv2Stride,
                                padding = 1)
        self.BatchNorm4 = NN.BatchNorm2d(Conv2OutputChannel)
        self.reluConv4 = NN.ReLU()
        
        self.MaxPoolConv2 = NN.MaxPool2d((2,2), stride = 2)
        self.DropOut2 = NN.Dropout2d(p=dropout)
        ########################################################################################## 
        #Output size at this point is 8x8x64
        self.Conv5 = NN.Conv2d(in_channels = Conv2OutputChannel, 
                                out_channels = Conv3OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv1Stride,
                                padding = 1)
        self.BatchNorm5 = NN.BatchNorm2d(Conv3OutputChannel)
        self.reluConv5 = NN.ReLU()
        
        self.Conv6 = NN.Conv2d(in_channels = Conv3OutputChannel, 
                                out_channels = Conv3OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv2Stride,
                                padding = 1)
        self.BatchNorm6 = NN.BatchNorm2d(Conv3OutputChannel)
        self.reluConv6 = NN.ReLU()
        
        self.MaxPoolConv3 = NN.MaxPool2d((2,2), stride = 2)
        self.DropOut3 = NN.Dropout2d(p=dropout)
        ##########################################################################################
        #Output size at this point is 4x4x128
        
        self.Conv7 = NN.Conv2d(in_channels = Conv3OutputChannel, 
                                out_channels = Conv4OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv1Stride,
                                padding = 1)
        self.BatchNorm7 = NN.BatchNorm2d(Conv4OutputChannel)
        self.reluConv7 = NN.ReLU()
        
        self.Conv8 = NN.Conv2d(in_channels = Conv4OutputChannel, 
                                out_channels = Conv4OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv2Stride,
                                padding = 1)
        self.BatchNorm8 = NN.BatchNorm2d(Conv4OutputChannel)
        self.reluConv8 = NN.ReLU()
        
        self.MaxPoolConv4 = NN.MaxPool2d((2,2), stride = 2)
        self.DropOut4 = NN.Dropout2d(p=dropout)
        
        ##########################################################################################
        #Output size at this point is 2x2x256
        self.FullyConn1 = NN.Linear(in_features = 1024, out_features=512)
        self.BatchNorm9 = NN.BatchNorm1d(512)
        self.reluFC1 = NN.ReLU()
        self.DropOut5 = NN.Dropout1d(p=dropout)
        
        self.FullyConn2 = NN.Linear(in_features = 512, out_features=256)
        self.BatchNorm10 = NN.BatchNorm1d(256)
        self.reluFC2 = NN.ReLU()
        self.DropOut6 = NN.Dropout1d(p=dropout)
        
        #Final Layer
        self.FullyConn3 = NN.Linear(in_features = 256, out_features = classes)
        self.SoftMax = NN.LogSoftmax(dim=1)
        
        
    def forward(self, input):
        
        #Block 1
        input = self.Conv1(input)
        #FirstConvLayerOutput = input
        input = self.BatchNorm1(input)
        input = self.reluConv1(input)
        input = self.Conv2(input)
        input = self.BatchNorm2(input)
        input = self.reluConv2(input)
        input = self.MaxPoolConv1(input)
        input = self.DropOut1(input)
        
        #Block 2
        input = self.Conv3(input)
        input = self.BatchNorm3(input)
        input = self.reluConv3(input)
        input = self.Conv4(input)
        input = self.BatchNorm4(input)
        input = self.reluConv4(input)
        input = self.MaxPoolConv2(input)
        input = self.DropOut2(input)
        
        #Block 3
        input = self.Conv5(input)
        input = self.BatchNorm5(input)
        input = self.reluConv5(input)
        input = self.Conv6(input)
        input = self.BatchNorm6(input)
        input = self.reluConv6(input)
        input = self.MaxPoolConv3(input)
        input = self.DropOut3(input)
        
        #Block 4
        input = self.Conv7(input)
        input = self.BatchNorm7(input)
        input = self.reluConv7(input)
        input = self.Conv8(input)
        input = self.BatchNorm8(input)
        input = self.reluConv8(input)
        input = self.MaxPoolConv4(input)
        input = self.DropOut4(input)
        
        #Block 3 FC Layers
        input = input.view(input.size(0), -1)
        
        input = self.FullyConn1(input)
        input = self.BatchNorm9(input)
        input = self.reluFC1(input)
        input = self.DropOut5(input)
        
        input = self.FullyConn2(input)
        input = self.BatchNorm10(input)
        input = self.reluFC2(input)
        input = self.DropOut6(input)
        
        input = self.FullyConn3(input)
        output = self.SoftMax(input)
        
        return output

"""###################################################################################"""

"""###################################################################################"""
"""The main function starts here which takes the input from the user and decides to perform training or testing of the model"""  
if __name__ =="__main__":
    #Check Device availability
    device = DeviceCheck()
    
    cnn = CIFARNet(numChannels = 3,classes = 10).to(device)
    summary(cnn,(3,32,32))
    print(cnn)