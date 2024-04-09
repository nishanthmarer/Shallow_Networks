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

x = torch.zeros((8,8,64))

"""###################################################################################"""   
"""This function creates the Neural Network structure for the CIFAR10 dataset"""
"""Batch Normalization and Dropout is added as a part of regularization process 
to help the model predict the test images better and increase the accuracy"""
class CIFARNet(NN.Module):
    def __init__(self, numChannels, classes):
        super(CIFARNet,self).__init__()
        #VGG Type Network
        #First Layer as per requirement is 
        #F = 5x5, S = 1, K = 32, Input size = 32x32, Output Size = 32x32x32
        self.Conv1_Main = NN.Conv2d(in_channels = numChannels, 
                                out_channels = Conv1OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv1Stride,
                                padding = 1)
        self.BatchNorm1_Main = NN.BatchNorm2d(Conv1OutputChannel)
        self.reluConv1_Main = NN.ReLU()
        
        self.Conv2_Main = NN.Conv2d(in_channels = Conv1OutputChannel, 
                                out_channels = Conv1OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv2Stride,
                                padding = 1)
        self.BatchNorm2_Main = NN.BatchNorm2d(Conv1OutputChannel)
        self.reluConv2_Main = NN.ReLU()
        
        self.MaxPoolConv1_Main = NN.MaxPool2d((2,2), stride = 2)
        self.DropOut1_Main = NN.Dropout2d(p=0.2)
        
        #Output at this point is 16x16x32 -> Stream 1
        self.Conv1_S1_1 = NN.Conv2d(in_channels = Conv1OutputChannel, 
                                out_channels = Conv1OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv1Stride,
                                padding = 1)
        self.BatchNorm1_S1_1 = NN.BatchNorm2d(Conv1OutputChannel)
        self.reluConv1_S1_1 = NN.ReLU()
        
        self.Conv2_S1_1 = NN.Conv2d(in_channels = Conv1OutputChannel, 
                                out_channels = Conv2OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv2Stride,
                                padding = 1)
        self.BatchNorm2_S1_1 = NN.BatchNorm2d(Conv2OutputChannel)
        self.reluConv2_S1_1 = NN.ReLU()
                
        self.MaxPoolConv1_S1 = NN.MaxPool2d((2,2), stride = 2)
        self.DropOut1_S1 = NN.Dropout2d(p=0.2)
        
        #Output at this point is 8x8x64 -> Stream 1
        ###############################################################################################################################################
        #Stream 2 start point from main of size 16x16x32
        self.Conv1_Main_2 = NN.Conv2d(in_channels = Conv1OutputChannel, 
                                out_channels = Conv2OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv1Stride,
                                padding = 1)
        self.BatchNorm1_Main_2 = NN.BatchNorm2d(Conv2OutputChannel)
        self.reluConv1_Main_2 = NN.ReLU()
        
        self.Conv2_Main_2 = NN.Conv2d(in_channels = Conv2OutputChannel, 
                                out_channels = Conv2OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv2Stride,
                                padding = 1)
        self.BatchNorm2_Main_2 = NN.BatchNorm2d(Conv2OutputChannel)
        self.reluConv2_Main_2 = NN.ReLU()
        
        self.MaxPoolConv1_Main_2 = NN.MaxPool2d((2,2), stride = 2)
        self.DropOut1_Main_2 = NN.Dropout2d(p=0.2)
        
        #Output at this point is 8x8x64
        #Start of Stream 2
        self.Conv1_S2_1 = NN.Conv2d(in_channels = Conv2OutputChannel, 
                                out_channels = Conv2OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv1Stride,
                                padding = 1)
        self.BatchNorm1_S2_1 = NN.BatchNorm2d(Conv2OutputChannel)
        self.reluConv1_S2_1 = NN.ReLU()
        
        self.Conv2_S2_1 = NN.Conv2d(in_channels = Conv2OutputChannel, 
                                out_channels = Conv2OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv2Stride,
                                padding = 1)
        self.BatchNorm2_S2_1 = NN.BatchNorm2d(Conv2OutputChannel)
        self.reluConv2_S2_1 = NN.ReLU()
                
        #Stream 1 + Stream2 = 8x8x64 + 8x8x64 = 8x8x128
        #self.fusion1 = torch.cat((x,x),2)
        self.fusion_Maxpool = NN.MaxPool2d((2,2), stride = 2)
        #Output at this point is 8x8x64 -> Stream 2 (Combine Stream 1 + Stream 2 = 8x8x64 + 8x8x64 = 8x8x128 and do maxpool so that it becomes 4x4x128)
        ###############################################################################################################################################
        #Stream 3 start point from main of size 8x8x64
        
        self.Conv1_Main_3 = NN.Conv2d(in_channels = Conv2OutputChannel, 
                                out_channels = Conv3OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv1Stride,
                                padding = 1)
        self.BatchNorm1_Main_3 = NN.BatchNorm2d(Conv3OutputChannel)
        self.reluConv1_Main_3 = NN.ReLU()
        
        self.Conv2_Main_3 = NN.Conv2d(in_channels = Conv3OutputChannel, 
                                out_channels = Conv3OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv2Stride,
                                padding = 1)
        self.BatchNorm2_Main_3 = NN.BatchNorm2d(Conv3OutputChannel)
        self.reluConv2_Main_3 = NN.ReLU()
        
        self.MaxPoolConv1_Main_3 = NN.MaxPool2d((2,2), stride = 2)
        self.DropOut1_Main_3 = NN.Dropout2d(p=0.2)
        
        #Output at this point is 4x4x128
        #Start of Stream 3
        self.Conv1_S3_1 = NN.Conv2d(in_channels = Conv3OutputChannel, 
                                out_channels = Conv3OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv1Stride,
                                padding = 1)
        self.BatchNorm1_S3_1 = NN.BatchNorm2d(Conv3OutputChannel)
        self.reluConv1_S3_1 = NN.ReLU()
        
        self.Conv2_S3_1 = NN.Conv2d(in_channels = Conv3OutputChannel, 
                                out_channels = Conv3OutputChannel, 
                                kernel_size = Conv2Filter_Size, 
                                stride = Conv2Stride,
                                padding = 1)
        self.BatchNorm2_S3_1 = NN.BatchNorm2d(Conv3OutputChannel)
        self.reluConv2_S3_1 = NN.ReLU()
        
        #Output at this point is 4x4x128 -> Stream 3
        #Combine this with stream 1 + stream 2 output i.e., 4x4x128 -> stream 3 + (stream 1 + stream 2) = 4x4x128 + 4x4x128 = 4x4x256
        
        #@Todo implement two fusion blocks, 
        #one block combines stream 1 and stream 2 
        #second block combines stream 3 and output of fusion block 1.
        #this is then given to the max pool for making it 2x2x256
        self.MaxPoolConv4 = NN.MaxPool2d((2,2), stride = 2)
        self.DropOut4 = NN.Dropout2d(p=0.2)
        
        ##########################################################################################
        #Output size at this point is 2x2x256
        self.FullyConn1 = NN.Linear(in_features = 1024, out_features=512)
        self.BatchNorm9 = NN.BatchNorm1d(512)
        self.reluFC1 = NN.ReLU()
        self.DropOut5 = NN.Dropout1d(p=0.2)
        
        self.FullyConn2 = NN.Linear(in_features = 512, out_features=256)
        self.BatchNorm10 = NN.BatchNorm1d(256)
        self.reluFC2 = NN.ReLU()
        self.DropOut6 = NN.Dropout1d(p=0.2)
        
        #Final Layer
        self.FullyConn3 = NN.Linear(in_features = 256, out_features = classes)
        self.SoftMax = NN.LogSoftmax(dim=1)
        
        
    def forward(self, input):
        
        #Block 1
        input = self.Conv1_Main(input)
        #FirstConvLayerOutput = input
        input = self.BatchNorm1_Main(input)
        input = self.reluConv1_Main(input)
        input = self.Conv2_Main(input)
        input = self.BatchNorm2_Main(input)
        input = self.reluConv2_Main(input)
        input = self.MaxPoolConv1_Main(input)
        input = self.DropOut1_Main(input)
        
        #Stream 1 16x16x32 stream
        input1 = self.Conv1_S1_1(input)
        input1 = self.BatchNorm1_S1_1(input1)
        input1 = self.reluConv1_S1_1(input1)
        
        input1 = self.Conv2_S1_1(input1)
        input1 = self.BatchNorm2_S1_1(input1)
        input1 = self.reluConv2_S1_1(input1)
                
        input1 = self.MaxPoolConv1_S1(input1)
        input1 = self.DropOut1_S1(input1)
        
        #at this point Stream 1 output is 8x8x64
        
        #second block of the main to convert from 16x16x32 -> 8x8x64
        input = self.Conv1_Main_2(input)
        input = self.BatchNorm1_Main_2(input)
        input = self.reluConv1_Main_2(input)
        
        input = self.Conv2_Main_2(input)
        input = self.BatchNorm2_Main_2(input)
        input = self.reluConv2_Main_2(input)
        
        input = self.MaxPoolConv1_Main_2(input)
        input = self.DropOut1_Main_2(input)
        
        #Stream 2 starts here and with an input of 8x8x64
        input2 = self.Conv1_S2_1(input)
        input2 = self.BatchNorm1_S2_1(input2)
        input2 = self.reluConv1_S2_1(input2)
        
        input2 = self.Conv2_S2_1(input2)
        input2 = self.BatchNorm2_S2_1(input2)
        input2 = self.reluConv2_S2_1(input2)
        
        #at this point the output of stream2 will be 8x8x64
        #here we need a fusion block to combine the input1 + input2 to make 8x8x128
        
        input3 = torch.cat((input1,input2),dim=1) #fusion block one1
        input3 = self.fusion_Maxpool(input3)
        #Also we need a maxpool so that we make it align with the steam 3 output of 4x4x128
        
        #Third block of the main to convert from 8x8x64 -> 4x4x128
        input = self.Conv1_Main_3(input)
        input = self.BatchNorm1_Main_3(input)
        input = self.reluConv1_Main_3(input)
        
        input = self.Conv2_Main_3(input)
        input = self.BatchNorm2_Main_3(input)
        input = self.reluConv2_Main_3(input)
        
        input = self.MaxPoolConv1_Main_3(input)
        input = self.DropOut1_Main_3(input)
        
        #stream 3 starts here with an input of 4x4x128
        input = self.Conv1_S3_1(input)
        input = self.BatchNorm1_S3_1(input)
        input = self.reluConv1_S3_1(input)
        
        input = self.Conv2_S3_1(input)
        input = self.BatchNorm2_S3_1(input)
        input = self.reluConv2_S3_1(input)
                
        #At this point the output will be 4x4x128 and this needs to be combined with the output of 
        #stream 1 + stream 2.
        #stream 3 + (stream 1 + stream 2 ) = 4x4x256
        input = torch.cat((input,input3),dim=1) #fusion block 2
        
        #we will apply max pool to the above output and get 2x2x256
        input = self.MaxPoolConv4(input)
        input = self.DropOut4(input)
        
        #FC Layers starts here
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