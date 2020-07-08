# SH_HRTF_Restoration
> Low-order Spherical Harmonic HRTF Restorationusing a Neural Network Approach

 *Correspondence: bt712@york.ac.uk (Ben)*

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
  * [torch (PyTorch)](https://pytorch.org/)
  * [visdom](https://github.com/facebookresearch/visdom)
* (Used PyCharm as IDE for this project)
* MATLAB (MATLAB 2019 and 2020 are both used in this project)

## Training data
Please download the training data from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing).

## Training a model
**Please make sure all the python packages have been correctly installed and training data has been downloaded from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing).**
1. Open `\HRTF_Restoration_01\Python_scripts\Training\training_HRTF_baseline.py` (or `\training_HRTF_proposed.py` or `\training_HRTF_bigger.py`)
2. Start visdom by running `python -m visdom.server` .
3. Update the folder path in line 31 (e.g. `file_loc = 'C:/Users/Admin/Downloads/HRTF_Restoration_01/Training_data/Time_aligned/'` <- Training data downloaded from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing)).
4. (If necessary) Change the number of epoch (line 341 in `training_HRTF_baseline.py` and `training_HRTF_proposed.py`, line 347 in `training_HRTF_bigger.py`), default 500.
5. Change the model save name, recommend to end with `.pt` (line 560 in `training_HRTF_baseline.py` and `training_HRTF_proposed.py`, line 566 in `training_HRTF_bigger.py`)
6. Run the script.
(Please ignore the `couldn't retrieve source code for container of type` error warning if showed up.)

## Evaluation
### Export results from trained model and save as .csv (Python)
**1st order SH interpolated SADIE subject 18, SADIE subject 19, SADIE subject 20 and Bernschutz KU100 HRTFs will be processed with the trained models.**
1. Open `\HRTF_Restoration_01\Python_scripts\Validate\export_result_baseline.py` (or `\export_result_proposed.py` or `\export_result_bigger.py`)
2. Update the folder path in line 217 in `export_result_baseline.py` and `export_result_proposed.py`, line 223 in `export_result_bigger.py` (e.g. `folder_loc = 'C:/Users/Admin/Downloads/HRTF_Restoration_01/Training_data/Time_aligned/'`
 <- Training data downloaded from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing)).
3. Run the script.
4. The output of the 4 HRTFs data will be save in 4 .csv files respectively **(Renaming the files is not recommanded, as it may causes extra modifications in the MATLAB scripts that will be used for analysis)**.
5. Please move those .csv files to your MATLAB directory (`\HRTF_Restoration_01\Matlab_scripts\model_ouput` is recommanded). 
*Extra info:*
*There are extra trained models provide in the `Models` folder, feel free to modify the `export_result_baseline.py` script and test them out.*
*`training_HRTF_08++_15_sparse.pt` = Baseline model* ***(default)***
*`training_HRTF_08++_16_sparse.pt` = Baseline model with weight decay*
*`training_HRTF_08++_13_sparse.pt` = Baseline model with drop out*
*`training_HRTF_08++_12_sparse.pt` = Baseline model with weight decay drop out* ***(propsed model)***
*`training_HRTF_08++_12_sparse.pt` = Baseline model with weight decay drop out and early stopped at 111 epoch*
*`training_HRTF_08++_12_sparse.pt` = Baseline model with weight decay drop out and early stopped at 111 epoch*
*`training_HRTF_08++_19_sparse.pt` = Baseline model with L1 loss*
*`training_HRTF_08++_20_sparse.pt` = Baseline model with MSE loss*
*`training_HRTF_08++_20_sparse.pt` = Baseline model without extra data from ARI, ITA and RIEC database*

***Instructions:***
1. *Update the model architecture in `class Net(nn.Module):` line 141 in `export_result_baseline.py`.*
2. *Update the nome of the saved model in line 209 (.pt is not needed, e.g. `model_name = 'training_HRTF_08++_16_sparse'`)*
3. *Run the script.*

*Please feel free to email **bt712@york.ac.uk** if there is any problem.*

### Evaluate with the PSD model (MATLAB)
1. Highlight all folders in `\HRTF_Restoration_01\Matlab_scripts`, right click `Add to Path` -> `Selected Folders and Subfolders`.
2. In `\HRTF_Restoration_01\Matlab_scripts\Evaluation\PspecModel` open `\compare_PSD_SADIE.m` for the SADIE HRTFs data and `\compare_PSD_Bern.m` for the Bernschutz KU100 HRTFs data.

**For `\compare_PSD_SADIE.m` with the SADIE subject 18, subject 19 and subject 20 HRTFs data**

3. Update the folder path from line 19 to 22. 
 - `hrtf_out` should be the .csv file exported from the Python sctipt.
 - `hrtf_in`, `hrtf_tar` and `angle_matched` should be the in the training data folder from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing)). Change the path before `/Time_aligned/` should work.
4. Run the function in the command window:
```
subject = '18';  % or  '19' or  '20', the subject number in SADIE exported data
model = 'training_HRTF_08++_12_sparse'; % the name of the saved model
[PSD_in_18_summary,PSD_out_18_summary, PSD_in_18, PSD_out_18] = compare_PSD_SADIE('18',model,1);
 ```
*Note about plot flag:*
* 0 = no plot*
* 1 = 3d plot*
* 2 = heatmap*
* 3 = both 3d plot and heat map*

**For `\compare_PSD_Bern.m` with the Bernschutz KU100 HRTFs data**

3. Update the folder path from line 17 to 20. 
 - `hrtf_out` should be the .csv file exported from the Python sctipt.
 - `hrtf_in`, `hrtf_tar` and `angle_matched` should be the in the training data folder from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing)). Change the path before `/Time_aligned/` should work.
4. Run the function in the command window:
```
model = 'training_HRTF_08++_12_sparse'; % the name of the saved model
[PSD_in_bern_summary,PSD_out_bern_summary, PSD_in_bern, PSD_out_bern] = compare_PSD_Bern(model,1);
```
*Note about plot flag:*
* *0 = no plot*
* *1 = 3d plot*
* *2 = heatmap*
* *3 = both 3d plot and heat map*

### Evaluate with the localisation models (MATLAB)
1. Highlight all folders in `HRTF_Restoration_01\Matlab_scripts`, right click `Add to Path` -> `Selected Folders and Subfolders`.
2. In `\HRTF_Restoration_01\Matlab_scripts\Evaluation\Localisation` open `\may_model_csv.m`.
3. Update the name of the saved model in line 1 (e.g. `model = 'training_HRTF_08++_12_sparse';`)

**For SADIE subject 18, subject 19 and subject 20 HRTFs data**

4. Update the folder path from line 4 to 8. **(Make sure they are not commented out)**
 - `subject` is the subject number in the SADIE database (18, 19 or 20 in this case)
 - `hrtf_out` should be the .csv file exported from the Python sctipt.
 - `hrtf_in`, `hrtf_tar` and `angle_matched` should be the in the training data folder from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing)). Change the path before `/Time_aligned/` should work.
5. Run the script

**For the Bernschutz KU100 HRTFs data**

4. Comment out line 4 to 8.
5. Uncomment and update the folder path from line 10 to 14. **(Make sure they are not commented out)**
 - `hrtf_out` should be the .csv file exported from the Python sctipt.
 - `hrtf_in`, `hrtf_tar` and `angle_matched` should be the in the training data folder from [Google Drive](https://drive.google.com/drive/folders/1eZiNmvomlguggppe_GQdkP89mM-CMjhy?usp=sharing)). Change the path before `/Time_aligned/` should work.
6. Run the script



## *Please feel free to email **bt712@york.ac.uk** if there is question about this project.*

