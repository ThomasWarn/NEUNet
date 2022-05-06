# NEUNet
A novel exponential U-Net, currently evaluating DLRSD

# Requirements
pip install requirements.txt

# Procedure to reproduce
1. Download DLRSD image & segmentation dataset from http://weegee.vision.ucmerced.edu/datasets/landuse.html and https://sites.google.com/view/zhouwx/dataset .
2. Place the respective extracted contents (agricultural, airplane, baseballdiamond...) in a respective folder next to the scripts. Name the folders "DLRSD" and "DLRSD_Segmented" respectively.
3. Execute 01_Prepare_Dataset.py
4. Execute 02_Train_Model.py 
