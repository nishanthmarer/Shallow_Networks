import os
import argparse
import torch as TH
import torchvision as TV
import matplotlib.pyplot as plt
import torch.nn as NN
import numpy as np
import cv2
from torch.utils.data import random_split
from DataLoader import createDataset,ImagePlotter
from datetime import date

import DataLoader_2
import ResNet
import VGGNet
import ShallowModel_3Streams_1Block
import ShallowModel_3Streams_1Block_Depth
import ShallowModel_3Streams_2Block
import ShallowModel_4Streams_2Block
import ShallowModel_ResNet_3Streams_1Block
import ShallowModel_ResNet_3Streams_1Block_Depth
import ShallowModel_ResNet_3Streams_2Block
import ShallowModel_ResNet_3Streams_2Block_Depth
import ShallowModel_ResNet_3Streams_2Block_Depth_Skip_Connection
import ShallowModel_ResNet_3Streams_3Block
import ShallowModel_ResNet_3Streams_3Block_Depth
import ShallowModel_ResNet_3Streams_3Block_Depth_Skip_Connection
######

"""Set the debug = 1 to print messages in the code else set to 0 to stop printing output"""
debug = 1
Batch_Size = 128
Epoch = 200
Learning_Rate = 1e-2
Weight_Deacy = 1e-4
Patience = 8
DataStoreLocCIFAR = "DataStore/CIFAR10"
CIFAR10_Image_Size = 32
ModelLocCIFAR = "model/CIFAR10.pth" 
CIFAR_Classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        self.dataset = dl.dataset
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
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
"""This function takes the dataset and the neural network structure and trains it. The training is done until the validation loss reduces
and once the loss stops reducing and starts increasing the patience counter increments until the limiter and prevents the model overfitting"""
def TrainAndValidateModel(cnn, ModelSaveLoc, trainingDataLoader, validationDataLoader, testingDataLoader, Optimizer, LossFunction, scheduler, device, Patience):
    
    TrainingStepSize = len(trainingDataLoader.dataset) // Batch_Size
    ValidationStepSize = len(validationDataLoader.dataset) // Batch_Size
    
    #Use the early stopping function to prevent overfitting of the model
    Best_Validation_Loss = np.inf
    Patience_Counter = 0
    
    print("Epoch \tTraining Loss \tTraining Accuracy% \tValidation Loss \tValidation Accuracy% \tTesting Loss \tTesting Accuracy%")
    #Start the model training now
    for ep in range(Epoch):
        #Set the model to training Mode
        cnn.train()
        
        TrainingLossPerEP = 0
        ValidationLossPerEP = 0
        TestLossPerEP = 0
        
        TrainingCorrectPredPerEP = 0
        ValidationCorrectPredPerEP = 0
        TestingCorrectPredPerEP = 0
        
        for (Image,Label_Val) in trainingDataLoader:
            
            Optimizer.zero_grad()
            
            (xData,yData) = (Image.to(device),Label_Val.to(device))
            ModelPrediction = cnn(xData)
            Loss = LossFunction(ModelPrediction,yData)
        
            Loss.backward()
            Optimizer.step()
            scheduler.step()
            
            TrainingLossPerEP += Loss.item()
            TrainingCorrectPredPerEP += (ModelPrediction.argmax(1) == yData).type(TH.float).sum().item()
         
        with TH.no_grad():
            cnn.eval()
            """Test on Validation Dataset"""
            for (Image,Label_Val) in validationDataLoader:
                
                (xData,yData) = (Image.to(device),Label_Val.to(device))
                ModelPrediction = cnn(xData)
                ValLoss = LossFunction(ModelPrediction,yData)
                
                ValidationLossPerEP += ValLoss.item() #this will sum upto the number of batches
                ValidationCorrectPredPerEP += (ModelPrediction.argmax(1) == yData).type(TH.float).sum().item() #This will sum upto number of images in the validation set
            
            """Test on Testing Dataset"""
            for (Image,Label_Val) in testingDataLoader:
                
                (xData,yData) = (Image.to(device),Label_Val.to(device))
                ModelPrediction = cnn(xData)
                TestLoss = LossFunction(ModelPrediction,yData)
                
                TestLossPerEP += TestLoss.item() #this will sum upto the number of batches
                TestingCorrectPredPerEP += (ModelPrediction.argmax(1) == yData).type(TH.float).sum().item() #This will sum upto number of images in the testing set

        #Calculate the Average Loss per Epoch
        AverageTrainingLoss = TrainingLossPerEP / TrainingStepSize
        AverageValidationLoss = ValidationLossPerEP / ValidationStepSize
        AverageTestingLoss = TestLossPerEP / len(testingDataLoader.dataset)
        
        #Calculate the Training and Validation accuracy
        TrainingCorrectPredPerEP = (TrainingCorrectPredPerEP / len(trainingDataLoader.dataset))*100.0
        ValidationCorrectPredPerEP = (ValidationCorrectPredPerEP / len(validationDataLoader.dataset))*100.0
        TestingCorrectPredPerEP = (TestingCorrectPredPerEP /  len(testingDataLoader.dataset))*100.0
        
        #Print the training progress and accuracy achieved
        print(f'{ep+1}\t{AverageTrainingLoss:.6f}\t{TrainingCorrectPredPerEP:.4f}\t\t\t{AverageValidationLoss:.6f}\t\t{ValidationCorrectPredPerEP:.4f}\t\t\t{AverageTestingLoss:.6f}\t{TestingCorrectPredPerEP:.4f}')
        
        #Add early stopping to prevent the model from overfitting to the training data
        if AverageValidationLoss < Best_Validation_Loss:
            Best_Validation_Loss = AverageValidationLoss
            Patience_Counter = 0
            TH.save(cnn.state_dict(), ModelSaveLoc)
            if debug: print("Validation Loss Improved. Saved model.")
        else:
            Patience_Counter += 1
            if debug: print("Early stopping counter: {}/{}".format(Patience_Counter,Patience))
            if Patience_Counter >= Patience:
                print("Stopped early. Best Validation Loss: {:.4f}".format(Best_Validation_Loss))
                break
        
                
"""###################################################################################"""
"""This function takes the trained model and tests it against the testing data set and displays the accuracy obtained"""
def TestModel(cnn, testingDataLoader, device):
    
    cnn.eval()
    with TH.no_grad():
        correct = 0
        for images, labels in testingDataLoader:
            test_output = cnn(images.to(device))
            pred_y = test_output.argmax(dim=1)
            correct += (pred_y == labels.to(device)).type(TH.float).sum().item()
        accuracy = (correct/len(testingDataLoader.dataset))*100.0
    
    print("The Accuracy of the model from the latest checkpoint is {:.4f}%".format(accuracy))

"""###################################################################################"""
"""This function takes the image given by the user and returns the prediction given by the model"""
def TestModelSingleInput(cnn, Image, device):
    cnn.eval()
    with TH.no_grad():
        
        test_output, img = cnn(Image.to(device))
        pred_y = test_output.argmax(dim=1)

        print("The Predicited Output is", CIFAR_Classes[pred_y.to("cpu").numpy()[0]])
    
    
"""###################################################################################"""
"""This function takes the input image and re-sizes it according to the CIFAR10 dataset size"""
def ReSizeImage(Image,ImageTypeSize):
    if Image.shape[0] > ImageTypeSize and Image.shape[1] > ImageTypeSize:
        Image = cv2.resize(Image, (ImageTypeSize,ImageTypeSize))
    return Image

"""#######################################################################################"""
"""This function selects a given model by name and creates an instance and returns it"""
def model_Selector(modelName):
    if modelName == "VGGNet":
        cnn = VGGNet.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ResNet":
        cnn = ResNet.ResNet9(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_3Streams_1Block":
        cnn = ShallowModel_3Streams_1Block.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_3Streams_1Block_Depth":
        cnn = ShallowModel_3Streams_1Block_Depth.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_3Streams_2Block":
        cnn = ShallowModel_3Streams_2Block.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)    
    elif modelName == "ShallowModel_4Streams_2Block":
        cnn = ShallowModel_4Streams_2Block.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_ResNet_3Streams_1Block":
        cnn = ShallowModel_ResNet_3Streams_1Block.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_ResNet_3Streams_1Block_Depth":
        cnn = ShallowModel_ResNet_3Streams_1Block_Depth.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_ResNet_3Streams_2Block":
        cnn = ShallowModel_ResNet_3Streams_2Block.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_ResNet_3Streams_2Block_Depth":
        cnn = ShallowModel_ResNet_3Streams_2Block_Depth.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_ResNet_3Streams_2Block_Depth_Skip_Connection":
        cnn = ShallowModel_ResNet_3Streams_2Block_Depth_Skip_Connection.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_ResNet_3Streams_3Block":
        cnn = ShallowModel_ResNet_3Streams_3Block.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_ResNet_3Streams_3Block_Depth":
        cnn = ShallowModel_ResNet_3Streams_3Block_Depth.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
    elif modelName == "ShallowModel_ResNet_3Streams_3Block_Depth_Skip_Connection":
        cnn = ShallowModel_ResNet_3Streams_3Block_Depth_Skip_Connection.CIFARNet(numChannels = 3,classes = len(trainingDataLoader.dataset.dataset.classes)).to(device)
        
    return cnn

"""###################################################################################"""
"""The main function starts here which takes the input from the user and decides to perform training or testing of the model"""  
if __name__ =="__main__":
    
    parser = argparse.ArgumentParser(description = 'Training on CIFAR10 using pytorch and CNN',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--mode', help='Train or Test the model', choices=['train','test']) 
    parser.add_argument('--modelPath',help='file location of model to be tested', type=str, default = None)
    parser.add_argument('--scheduler', help='Select a scheduler',choices=['OneCycleLR','CyclicLR'], default='OneCycleLR')
    parser.add_argument('--model', help='Select Models', choices=['ResNet','VGGNet',
                                                                  'ShallowModel_3Streams_1Block','ShallowModel_3Streams_1Block_Depth',
                                                                  'ShallowModel_3Streams_2Block','ShallowModel_4Streams_2Block',
                                                                  'ShallowModel_ResNet_3Streams_1Block','ShallowModel_ResNet_3Streams_1Block_Depth',
                                                                  'ShallowModel_ResNet_3Streams_2Block','ShallowModel_ResNet_3Streams_2Block_Depth',
                                                                  'ShallowModel_ResNet_3Streams_2Block_Depth_Skip_Connection','ShallowModel_ResNet_3Streams_3Block',
                                                                  'ShallowModel_ResNet_3Streams_3Block_Depth','ShallowModel_ResNet_3Streams_3Block_Depth_Skip_Connection'], default = 'ShallowModel_3Streams_1Block') 
    parser.add_argument('--dataLoader', help='Data Loader Type', choices=['Data1','Data2'], default='Data1')
    parser.add_argument('--TestImage', help='Test image for model testing', default = None)
    args = parser.parse_args()
    TestImage = ""
    train = False
    test = False
    if args.mode == "train":    
        DataSetType = "CIFAR10"
        DataStoreLoc = DataStoreLocCIFAR
        train = True
    elif args.mode == "test":
        TestImage = args.input
        DataSetType = "CIFAR10"
        if os.path.isfile(TestImage):
            if debug: print('Image File exists')
            test = True
        else:
            print('Image File does not exist')
            exit(0)
    else:
        print('The input is not right choice.. Exiting!')
        exit(0)
    
    #Check Device availability
    device = DeviceCheck()
    
    if train:
        CodePath = os.path.dirname(__file__)
        DataLocFullPath = os.path.join(CodePath, DataStoreLoc)
        
        """Create the DataStore folder is not exist"""
        os.makedirs(DataLocFullPath, exist_ok=True)
        
        if args.dataLoader == "Data1":
            trainingDataLoader,validationDataLoader,testingDataLoader = createDataset(DataLocFullPath,DataSetType, Batch_Size, CodePath)
        elif args.dataLoader == "Data2":
            trainingDataLoader,validationDataLoader,testingDataLoader = DataLoader_2.createDataset(DataLocFullPath,DataSetType, Batch_Size, CodePath)
        else:
            print("wrong input")
            exit(0)
            
        trainingDataLoader = DeviceDataLoader(trainingDataLoader, device)
        validationDataLoader = DeviceDataLoader(validationDataLoader, device)
        testingDataLoader = DeviceDataLoader(testingDataLoader, device)
        
        print("##########################################################################")
        print("Model Name: ",args.model)
        print("Batch Size: ",Batch_Size)
        print("Epoch: ",Epoch)
        print("Learning Rate: ",Learning_Rate)
        print("Weight Deacy: ",Weight_Deacy)
        print("DataSet loader: ", args.dataLoader)
        print("Schedular: ", args.scheduler)
        print("##########################################################################")
        
        if DataSetType == "CIFAR10":
            cnn = model_Selector(args.model)
        
        if debug: print(cnn)
    
        #Now we have the model, we need an optimizer and loss function
        """Weight Decay or Weight regularization is added as a part of regularization process to increase
        the accuray of the model predicition"""
        #Optimizer = TH.optim.Adam(cnn.parameters(),lr = Learning_Rate, weight_decay=Weight_Deacy)
        Optimizer = TH.optim.SGD(cnn.parameters(),lr = Learning_Rate, weight_decay=Weight_Deacy)
        if args.scheduler == "OneCycleLR":
            scheduler = TH.optim.lr_scheduler.OneCycleLR(Optimizer, Learning_Rate, epochs=Epoch, 
                                                     steps_per_epoch=len(trainingDataLoader))
        elif args.scheduler == "CyclicLR":
            scheduler = TH.optim.lr_scheduler.CyclicLR(Optimizer, base_lr = (Learning_Rate/10.0), max_lr = Learning_Rate, step_size_up = len(trainingDataLoader)//2, 
                                                       mode = "triangular")
            
        LossFunction = NN.CrossEntropyLoss()
        
        """Create the model folder is not exist"""
        os.makedirs(CodePath+"/model", exist_ok=True)
        current_Date = date.today().strftime('%Y-%m-%d')
        
        ModelPath = os.path.join(CodePath, "model/"+args.model+"_"+current_Date+".pth")
        TrainAndValidateModel(cnn, ModelPath, trainingDataLoader, validationDataLoader, testingDataLoader, Optimizer, LossFunction, scheduler, device, Patience)
        
        cnn.load_state_dict(TH.load(ModelPath))
        print("\nTesting Final Model on Testing DataSet")
        TestModel(cnn,testingDataLoader, device)
    
    elif test:
        print("Testing the model for a single image")
        img = cv2.imread(TestImage)
        CodePath = os.path.dirname(__file__)
        Converter = TV.transforms.Compose([TV.transforms.ToPILImage(),TV.transforms.ToTensor()])
        
        if DataSetType == "CIFAR10":
            cnn = model_Selector(args.model)
            
        img = ReSizeImage(img,CIFAR10_Image_Size)
        ModelPath = args.modelPath
        if ModelPath is not None and os.path.exists(ModelPath):
            cnn.load_state_dict(TH.load(ModelPath))
        else:
            print("Model file does not exists")
                
        colorimg = Converter(img)
        ImageFinal = colorimg.reshape(1,3,CIFAR10_Image_Size,CIFAR10_Image_Size)
            
        TestModelSingleInput(cnn, ImageFinal, device)

"""File Ends here"""