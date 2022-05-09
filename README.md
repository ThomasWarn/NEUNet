# NEUNet
A novel exponential U-Net, currently evaluating DLRSD

# Requirements
pip install requirements.txt

A GPU with 8GB+ of VRAM

32GB+ of system RAM

# Procedure to reproduce
1. Download DLRSD image & segmentation dataset from http://weegee.vision.ucmerced.edu/datasets/landuse.html and https://sites.google.com/view/zhouwx/dataset .
2. Place the respective extracted contents (agricultural, airplane, baseballdiamond...) in a respective folder next to the scripts. Name the folders "DLRSD" and "DLRSD_Segmented" respectively.

    Folder Structure & 'dir /b /s' should look like
```
Project\01_Prepare_Dataset.py
Project\02_Train_Model.py
Project\03_Generate_Verification_Data.py
Project\04_Test_With_Custom_Input.py
Project\DLRSD
	Project\DLRSD\agricultural
	Project\DLRSD\airplane
	Project\DLRSD\baseballdiamond
	...
Project\DLRSD_Segmented
	Project\DLRSD_Segmented\description.pdf
	Project\DLRSD_Segmented\Images
	Project\DLRSD_Segmented\legend.png
	Project\DLRSD_Segmented\multi-labels.xlsx
	Project\DLRSD_Segmented\Images\agricultural
	Project\DLRSD_Segmented\Images\airplane
	Project\DLRSD_Segmented\Images\baseballdiamond
	...
Project\logs
```

3. Execute 01_Prepare_Dataset.py.
4. Execute 02_Train_Model.py.
5. Modify 03_Generate_Verification_Data.py to contain the SGD-optimized model.
6. Execute 03_Generate_Verification_Data.py.


If exclusively evaluating the model performance on a seperate dataset, a trained model can be downloaded from https://drive.google.com/file/d/1Jnz3rS8emz8S1C8yHqdjW48cxwBzARNE/view?usp=sharing

This model can be tested using 04_Test_With_Custom_Input.py. It is recommended to only use imagery from DLSRD as inputs although other data can be used as input. The network appears to be particulary sensetive to shadows so directly overhead lighting is critical. Also image scale should be identical to the DLSRD training data. Pretty much just make your images look similar to the DLSRD and it should work acceptably.
