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


