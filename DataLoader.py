"""///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Name: Nishant Marer Prabhu, Zhijun Hu, Tejal Patel
FileName: DataLoader.py
Course: Computer Vision
Description: This file is responsible for downloading the dataset, creates the batches according to the specified batch size, performs data augmentation
            and returns the training, testing and validation dataset. It also has function to plot the images, check GPU exists
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"""
import os
import argparse
import torch as TH
import torchvision as TV
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import random_split

######
"""Set the debug = 1 to print messages in the code else set to 0 to stop printing output"""
debug = 1
#Global Settings
DataStoreLocCIFAR = "DataStore\\CIFAR10"
CIFAR10_Image_Size = 32

CIFAR_Classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


#Plotting Details
Num_Row_Image = 6
Num_Column_Image = 6
Num_DataPlot_Row = 10
Num_DataPlot_Col = 10

"""###################################################################################"""
"""This function checks if the GPU is available in the machine"""
def DeviceCheck():
    #Device Check
    device = "NULL"

    if TH.cuda.is_available():
        if debug: print('GPU Available')
        device = TH.device("cuda")
    else:
        if debug: print('No GPU using CPU Usage Warning!!')
        device = TH.device("cpu")
        
    return device

    
"""###################################################################################"""
"""This function plots the images in the dataset in a 6x6 format"""
def ImagePlotter(TestingDataSet,DataSetType,ImageSaveLoc):
    
    #Plot one set of Images to Display
    DataPlot = plt.figure(figsize=(Num_DataPlot_Row, Num_DataPlot_Col))
    for i in range(1, Num_Row_Image * Num_Column_Image + 1):
        random_Index = TH.randint(len(TestingDataSet), size=(1,)).item()
        image, label = TestingDataSet[random_Index]
        DataPlot.add_subplot(Num_Row_Image, Num_Column_Image, i)
        plt.title(label)
        plt.axis("off")
        if DataSetType == "CIFAR10":
            image = image / 2 + 0.5
            plt.imshow(np.transpose(image, (1,2,0)).squeeze(), cmap="gray")
        else:
            plt.imshow(image.squeeze(), cmap="gray")
    if debug: plt.show()
    
    if DataSetType == "CIFAR10":
        plt.savefig(ImageSaveLoc+'/CIFAR_Sample_Image.png')

"""###################################################################################"""
"""This function is responsible for downloading the dataset as per the argument type 'DataSetType' and then further perform transformation
as required for improving the accuracy of the model predicition"""
def createDataset(DataLocFullPath, DataSetType, Batch_Size, ImageSaveLoc):

    """Data Augmentation for the CIFAR dataset is done as a part of regularization techniques to increase the accuray of the model"""
    Transformation = TV.transforms.Compose([TV.transforms.Resize((CIFAR10_Image_Size,CIFAR10_Image_Size)),TV.transforms.ToTensor(), TV.transforms.Normalize((0.5),(0.5))])
    Transformation_Train = TV.transforms.Compose([TV.transforms.Resize((CIFAR10_Image_Size,CIFAR10_Image_Size)),
                                                  TV.transforms.RandomHorizontalFlip(),
                                                  TV.transforms.RandomRotation(12),
                                                  TV.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                                  TV.transforms.ToTensor(), 
                                                  TV.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    TrainingDataSet = TV.datasets.CIFAR10(root= DataLocFullPath, train = True, transform = Transformation_Train, download = True)
    TestingDataSet = TV.datasets.CIFAR10(root= DataLocFullPath, train = False, transform = Transformation, download = True)

    #Plot the Images
    ImagePlotter(TestingDataSet,DataSetType, ImageSaveLoc)
    
    #Create Training, Validation and Test Datasets
    numTrainSamples = int(len(TrainingDataSet) * 0.75)
    numValSamples = int(len(TrainingDataSet) * 0.25)
    (trainDataSplit, valDataSplit) = random_split(TrainingDataSet,[numTrainSamples, numValSamples], generator=TH.Generator().manual_seed(42))

    #Load the Images into a DataLoader for easy access to images
    trainingDataLoader = TH.utils.data.DataLoader(trainDataSplit, batch_size = Batch_Size, shuffle = True)
    validationDataLoader = TH.utils.data.DataLoader(valDataSplit, batch_size = Batch_Size, shuffle = True)
    testingDataLoader = TH.utils.data.DataLoader(TestingDataSet, batch_size = 1, shuffle = True)
    #testingDataLoader = TH.utils.data.DataLoader(TestingDataSet, batch_size = Batch_Size, shuffle = True)
    
    return trainingDataLoader,validationDataLoader,testingDataLoader

"""###################################################################################"""
"""This function is used for printing the output of the first Convolution layer in the NN"""
def PrintIntermediateResult(ImageIntermediate, Conv1OutputChannel,Conv1LayerOutSizeCIFAR, DataType, ImageSaveLoc):
    j = 0
    if DataType == "CIFAR10":
        img = ImageIntermediate.reshape(Conv1OutputChannel,Conv1LayerOutSizeCIFAR,Conv1LayerOutSizeCIFAR)
    
    #Plot one set of Images to Display
    DataPlot = plt.figure(figsize=(10, 10))
    for i in range(1, Num_Row_Image * Num_Column_Image + 1):
       
        image = img[j].to('cpu')
        DataPlot.add_subplot(Num_Row_Image, Num_Column_Image, i)
        
        plt.axis("off")
        plt.imshow(image.squeeze(), cmap="gray")
        j+=1
        if(j>(Conv1OutputChannel-1)):
            break
    if debug: plt.show()
    if DataType == "CIFAR10":
        plt.savefig(ImageSaveLoc+'/CONV_rslt_cifar.png')

"""###################################################################################"""
"""This function takes an input image and checks if the image is color or gray scale by
checking the number of channels and the content of the R G B values"""
def IsGray(Image):
    
    if len(Image.shape) < 3: 
        return True
    if Image.shape[2]  == 1: 
        return True
    B,G,R = Image[:,:,0], Image[:,:,1], Image[:,:,2]
    if (B==G).all() and (B==R).all(): 
        return True
    return False

"""###################################################################################"""
"""This function takes the input image and re-sizes it according to the MNIST or CIFAR10 dataset size"""
def ReSizeImage(Image,ImageTypeSize):
    if Image.shape[0] > ImageTypeSize and Image.shape[1] > ImageTypeSize:
        Image = cv2.resize(Image, (ImageTypeSize,ImageTypeSize))
    return Image
        
"""###################################################################################"""
"""The main function starts here which takes the input from the user and decides to perform training or testing of the model"""  
if __name__ =="__main__":
    
    #parser = argparse.ArgumentParser(description = 'Training on two dataset namely CIFAR10 using pytorch and CNN',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--cifar', help='CIFAR10 Dataset', action = 'store_true')
    #args = parser.parse_args()
    Batch_Size = 32
    DataSetType = "CIFAR10"
    DataStoreLoc = DataStoreLocCIFAR
        
    CodePath = os.path.dirname(__file__)
    DataLocFullPath = os.path.join(CodePath, DataStoreLoc)
    
    """Create the DataStore folder is not exist"""
    os.makedirs(DataLocFullPath, exist_ok=True)
    
    trainingDataLoader,validationDataLoader,testingDataLoader = createDataset(DataLocFullPath,DataSetType, Batch_Size, CodePath)
        
"""File Ends here"""