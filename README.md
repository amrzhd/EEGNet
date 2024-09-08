# EEGNet for Motor Imagery Classification

## Introduction
Welcome to the EEGNet for Motor Imagery Classification repository! This project focuses on implementing a convolutional neural network (CNN) model based on the EEGNet architecture for classifying motor imagery tasks using electroencephalography (EEG) data. The model is designed to classify between four different motor imagery classes: Left Hand, Right Hand, Foot, and Tongue.

## Dataset
The EEG data used in this project is sourced from the [BCI Competition IV 2a](http://www.bbci.de/competition/iv/#dataset2a) dataset. Key characteristics of the dataset include:
- 9 Participants
- 22 Ag/AgCl Electrodes
- Sampling Frequency: 250 Hz
- Epoched Data: [2, 6] seconds
- Frequency Range: 0.5 - 100 Hz
- Band Pass Filtered: 4 - 40 Hz (Using IIR Filter)
- 4 Classes: Left Hand, Right Hand, Foot, Tongue
- Both sessions of data (T & E) were used!
### List of Events
The following events are annotated in the dataset:
- '1023': 1 Rejected trial
- '1072': 2 Eye movements
- '276': 3 Idling EEG (eyes open)
- '277': 4 Idling EEG (eyes closed)
- '32766': 5 Start of a new run
- '768': 6 Start of a trial
- '769': 7 Cue onset **Left** (class 1) : 0 
- '770': 8 Cue onset **Right** (class 2) : 1
- '771': 9 Cue onset **Foot** (class 3) : 2 
- '772': 10 Cue onset **Tongue** (class 4): 3

Events 7, 8, 9, and 10 were chosen for the classification.

## Model
The EEGNet model architecture used in this project is detailed below:
- **Block 1:** Two sequential convolutional steps are performed. First, F1 2D convolutional filters of size (1, 32) are applied to capture frequency information at 2Hz and above. Then, a Depthwise Convolution of size (C, 1) is used to learn a spatial filter. Batch Normalization and ELU nonlinearity are applied, followed by Dropout for regularization. An average pooling layer is used for dimensionality reduction.
- **Block 2:** Separable Convolution is used, followed by Pointwise Convolutions. Average pooling is used for dimension reduction.
- **Classification Block:** Features are passed directly to a softmax classification with N units, where N is the number of classes in the data.

For further details, refer to the original [EEGNet implementation](https://github.com/vlawhern/arl-eegmodels/tree/master).

### Training Setup
- Optimizer: Adam
- Batch size: 64
- Epochs: 500
- Learning Rate: 0.001
- Loss Function: Cross Entropy

## Requirements
To run the code in this repository, make sure you have the following dependencies installed:
- Python == 3.7 or 3.8
- PyTorch == 2.3 (verified working with 2.0 - 2.3, both for CPU and GPU)
- torch-summary == 1.4.5
- mne >= 0.17.1
- scikit-learn >= 0.20.1
- matplotlib >= 2.2.3

## References
If you use this code or the EEGNet model architecture in your work, please cite the original paper of the orignal model:

[V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces," Journal of Neural Engineering, vol. 15, no. 5, p. 056013, 2018.](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c)

If you used the dataset in your work, please cite the original paper of it:

[C. Brunner, R. Leeb, G. Müller-Putz, A. Schlögl, and G. Pfurtscheller, "BCI Competition 2008–Graz data set A," Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology, vol. 16, pp. 1-6, 2008.](https://lampz.tugraz.at/~bci/database/001-2014/description.pdf)
