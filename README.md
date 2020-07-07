# SH_HRTF_Restoration
> Low-order Spherical Harmonic HRTF Restorationusing a Neural Network Approach

Spherical harmonic (SH) interpolation is a commonly used method to spatially up-sample sparse Head Related Transfer Function (HRTF) datasets to denser HRTF datasets. However, depending on the number of sparse HRTF measurements and SH order, this process can introduce distortions in high frequency representation of the HRTFs. This paper investigates whether it is possible to restore some of the distorted high frequency HRTF components using machine learning algorithms. A combination of Convolutional Auto-Encoder (CAE) and Denoising Auto-Encoder (DAE) models is proposed to restore the high frequency distortion in SH interpolated HRTFs. Results are evaluated using both Perceptual Spectral Difference (PSD) and localisation prediction models, both of which demonstrate significant improvement after the restoration process.

## Hardware requirement
* Windows or Linux (Only tested with Windows 10)
* Nvidia CUDA-Enabled Graphics card (Tested with RTX 2080 Ti and Quadro P4000) 
* 16GB of RAM (Tested with 32GB or above)
* 18GB of Memory (For the training data and scripts)
* (Only tested on Intel CPU) 

## Software requirement
* Python 3.6 (Tested on Python 3.6.7)
* Python package: 
  * matplotlib
  * numpy
  * pandas
  * torch (PyTorch) - https://pytorch.org/
  * visdom - https://github.com/facebookresearch/visdom
* (Used PyCharm as IDE for this project)
* MATLAB (MATLAB 2019 and 2020 are both used in this project)

## Training data
Please download the training data from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing).

## Training a model
Please make sure all the python packages have been correctly installed and downloaded the training data from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing)..
1. Open `HRTF_Restoration_01\Python_scripts\Training\training_HRTF_baseline.py` (or `training_HRTF_proposed.py` or `training_HRTF_bigger.py`)
2. Start visdom by running `python -m visdom.server` .
3. Relocate folder path in line 31 (e.g. `file_loc = 'C:/Users/Admin/Downloads/HRTF_Restoration_01/Training_data/Time_aligned/' <- Training data folder `
4. (If necessary) Change the number of epoch (line 341 in training_HRTF_baseline.py and training_HRTF_proposed.py, line 347 in training_HRTF_bigger.py), default 500.
5. Change the model save name, recommend to end with `.pt` (line 560 in training_HRTF_baseline.py and training_HRTF_proposed.py, line 566 in training_HRTF_bigger.py)
6. Run the script.
(Please ignore the `couldn't retrieve source code for container of type` error warning if showed up.)

## Evaluation
