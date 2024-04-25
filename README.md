# Shallow Network

This README document provides the necessary instructions for training and testing the Shallow Network using the provided datasets and pre-trained models. Please ensure your system meets the following prerequisites before proceeding.

## Prerequisites

Ensure your environment is set up according to these specifications to avoid any compatibility issues:

- **Python Version**: 3.12.1
- **PyTorch Version**: 2.2.2+cu121
- **CUDA Version**: 12.1

These dependencies can be installed using `pip` or `conda`, based on your preference and existing setup.

The code also requires certain folders to be present, where it saves the model and stores the dataset.

- **DataStore/CIFAR10**: This folder will contain the location of the dataset
- **model**: This folder will contain all the trained models

Creation of these folders are vital for functioning of the scripts.

## Training Commands

You can initiate the training process using either SLURM or directly through a Python script on a local machine.

### SLURM Command

To submit a training job to a SLURM managed cluster, use the following command:

```sbatch train_sbatch.script --mode train --scheduler OneCycleLR --model ShallowModel_3Streams_1Block --dataLoader Data2```

The above command will train the "ShallowModel_3Streams_1Block" model, replace its name with any other model in the file list.

### Python Training Command

Alternatively, train the model directly using Python with the following command:

```python Training.py --mode train --scheduler OneCycleLR --model ShallowModel_3Streams_1Block --dataLoader Data2```

## Testing Command

After training, evaluate the model's performance on test images using the command below. Make sure to replace the placeholders with actual values.

```python Training.py --mode test --modelPath <path_to_model> --model <model_Name> --TestImage <imgLocation>```

### Parameters Description

- `--mode test`: Runs the script in test mode.
- `--modelPath <path_to_model>`: Specifies the file path to your trained model. Replace `<path_to_model>` with the actual path.
- `--model <model_Name>`: Indicates the model to use for testing. Replace `<model_Name>` with the name of your model.
- `--TestImage <imgLocation>`: Provides the path to the image file for testing. Replace `<imgLocation>` with the path to your test image.

## Additional Information

Ensure that CUDA is enabled if you are using GPU acceleration. This setup requires CUDA version 12.1 or higher, which should be compatible with your PyTorch installation. For any issues or further assistance, refer to the documentation for Python, PyTorch, CUDA, or contact your system administrator for SLURM configurations.
